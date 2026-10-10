/*
 * join_order_sem.c — TriDB semantic-operator placement (Figure-1 plan P4, ADR-0026).
 *
 * ADDITIVE to the FROZEN decision core (join_order.c) and the cost path (join_order_cost.c):
 * nothing here reads or writes LegStats / TriJoinOrder. Three pure-arithmetic SQL functions:
 *
 *   tridb_place_sem(sels, costs, hits, cache_cost)        -> int[]   predicate order
 *   tridb_sem_plan_cost(sels, costs, hits, ord, n_in, cc) -> float8  expected cost of an order
 *   tridb_sem_batch(k_remaining, sel_hat, min_b, max_b)   -> int     streaming FETCH size
 *
 * MODEL (classic expensive-predicate ordering, Hellerstein & Stonebraker). Predicate i has
 * selectivity sel_i, per-row LLM cost cost_i, cache hit rate hit_i; a cache hit costs cache_cost.
 *   c_eff_i = cost_i * (1 - hit_i) + cache_cost * hit_i
 *   rank_i  = (1 - sel_i) / c_eff_i            (c_eff_i == 0 -> +Infinity: free predicates first)
 * Applying predicates in DESCENDING rank minimises n_in * sum_pos c_eff[ord[pos]] * prod_{q<pos}
 * sel[ord[q]] for independent predicates. Ties break by ascending input index (stable).
 *
 * BIT-IDENTITY CONTRACT with src/planner/join_order_sem_ref.py (the executable spec): every
 * float8 result is produced by the SAME sequence of IEEE-754 binary64 operations as the Python,
 * one operation per statement, evaluated left to right:
 *   c_eff:  a = 1.0 - hit;  b = cost * a;  d = cache_cost * hit;  c_eff = b + d
 *   rank:   (1.0 - sel) / c_eff
 *   cost:   acc = 0.0; surv = 1.0; for pos: acc = acc + c_eff[j] * surv; surv = surv * sel[j];
 *           result = n_in * acc
 *   batch:  s = max(sel_hat, 1e-6); v = ceil((double) k / s); clamp to [min_batch, max_batch]
 * The Makefile strips -ffast-math (module-wide) and compiles THIS file with -ffp-contract=off,
 * so no a*b+c is fused into an FMA (GCC's default on ARM64/GX10 would otherwise fuse and break
 * parity with Python, which never fuses).
 *
 * Pure arithmetic, no MSVBASE symbols: builds against PG 13.4 (fork) and stock PG 16/17 headers.
 * No LLM call happens here (ADR-0026 §2): these are planner-side cost/placement only.
 */

#include "postgres.h"

#include <math.h>

#include "catalog/pg_type.h"	/* FLOAT8OID, INT4OID */
#include "fmgr.h"
#include "utils/array.h"

#define TRIDB_SEM_MAX_PREDS 64
#define TRIDB_SEM_SEL_FLOOR 1e-6

/* ---- array helpers --------------------------------------------------------- */

/*
 * Deconstruct a 1-D (or empty) float8[] into a palloc'd double array. ERRORs on NULL elements or
 * on more than one dimension. Returns the element count in *n.
 */
static double *
sem_get_float8_array(ArrayType *arr, const char *fname, const char *argname, int *n)
{
	Datum	   *elems;
	bool	   *nulls;
	int			nelems;
	double	   *out;
	int			i;

	if (ARR_NDIM(arr) > 1)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("%s: %s must be a one-dimensional array", fname, argname)));
	if (array_contains_nulls(arr))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("%s: %s must not contain NULLs", fname, argname)));

	deconstruct_array(arr, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, TYPALIGN_DOUBLE,
					  &elems, &nulls, &nelems);
	out = (double *) palloc((nelems > 0 ? nelems : 1) * sizeof(double));
	for (i = 0; i < nelems; i++)
		out[i] = DatumGetFloat8(elems[i]);
	*n = nelems;
	return out;
}

static int32 *
sem_get_int4_array(ArrayType *arr, const char *fname, const char *argname, int *n)
{
	Datum	   *elems;
	bool	   *nulls;
	int			nelems;
	int32	   *out;
	int			i;

	if (ARR_NDIM(arr) > 1)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("%s: %s must be a one-dimensional array", fname, argname)));
	if (array_contains_nulls(arr))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("%s: %s must not contain NULLs", fname, argname)));

	deconstruct_array(arr, INT4OID, sizeof(int32), true, TYPALIGN_INT,
					  &elems, &nulls, &nelems);
	out = (int32 *) palloc((nelems > 0 ? nelems : 1) * sizeof(int32));
	for (i = 0; i < nelems; i++)
		out[i] = DatumGetInt32(elems[i]);
	*n = nelems;
	return out;
}

/* ---- validation (mirrors join_order_sem_ref._validate) ---------------------- */

static void
sem_invalid(const char *fname, const char *detail)
{
	ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("%s: %s", fname, detail)));
}

/*
 * Validate the predicate arrays and cache_cost. NaN fails every range test below because each
 * test is written as !(lo <= x && x <= hi) — a NaN comparison is false. Infinity is rejected for
 * costs (inf * 0 at hit == 1 would be NaN).
 */
static void
sem_validate(const char *fname, const double *sels, int ns, const double *costs, int nc,
			 const double *hits, int nh, double cache_cost)
{
	int			i;

	if (ns != nc || ns != nh)
		sem_invalid(fname, "sels, costs and hits must have equal lengths");
	if (ns > TRIDB_SEM_MAX_PREDS)
		sem_invalid(fname, "at most 64 predicates are supported");
	if (!(cache_cost >= 0.0) || isinf(cache_cost))
		sem_invalid(fname, "cache_cost must be finite and >= 0");
	for (i = 0; i < ns; i++)
	{
		if (!(sels[i] >= 0.0 && sels[i] <= 1.0))
			sem_invalid(fname, "every sel must be in [0, 1]");
		if (!(costs[i] >= 0.0) || isinf(costs[i]))
			sem_invalid(fname, "every cost must be finite and >= 0");
		if (!(hits[i] >= 0.0 && hits[i] <= 1.0))
			sem_invalid(fname, "every hit must be in [0, 1]");
	}
}

/* ---- arithmetic core (one IEEE op per statement; see header) ---------------- */

static double
sem_c_eff(double cost, double hit, double cache_cost)
{
	double		miss = 1.0 - hit;
	double		llm = cost * miss;
	double		cached = cache_cost * hit;

	return llm + cached;
}

static double
sem_rank(double sel, double c_eff)
{
	double		gain;

	if (c_eff == 0.0)
		return (double) INFINITY;
	gain = 1.0 - sel;
	return gain / c_eff;
}

/*
 * Stable descending-rank order: insertion sort (n <= 64) that moves an element left only past a
 * STRICTLY smaller rank, so equal ranks keep ascending input index.
 */
static void
sem_order(const double *rank, int n, int32 *ord)
{
	int			i,
				j;

	for (i = 0; i < n; i++)
	{
		int32		cur = (int32) i;

		j = i;
		while (j > 0 && rank[ord[j - 1]] < rank[cur])
		{
			ord[j] = ord[j - 1];
			j--;
		}
		ord[j] = cur;
	}
}

/* ---- SQL surface ------------------------------------------------------------ */

/*
 * tridb_place_sem(sels float8[], costs float8[], hits float8[], cache_cost float8 DEFAULT 0.0)
 *   RETURNS int[]  — 0-based predicate indices, descending rank, ties by ascending index.
 */
PG_FUNCTION_INFO_V1(tridb_place_sem_sql);
Datum
tridb_place_sem_sql(PG_FUNCTION_ARGS)
{
	const char *fname = "tridb_place_sem";
	int			ns,
				nc,
				nh,
				i;
	double	   *sels = sem_get_float8_array(PG_GETARG_ARRAYTYPE_P(0), fname, "sels", &ns);
	double	   *costs = sem_get_float8_array(PG_GETARG_ARRAYTYPE_P(1), fname, "costs", &nc);
	double	   *hits = sem_get_float8_array(PG_GETARG_ARRAYTYPE_P(2), fname, "hits", &nh);
	double		cache_cost = PG_GETARG_FLOAT8(3);
	double		rank[TRIDB_SEM_MAX_PREDS];
	int32		ord[TRIDB_SEM_MAX_PREDS];
	Datum		out[TRIDB_SEM_MAX_PREDS];

	sem_validate(fname, sels, ns, costs, nc, hits, nh, cache_cost);

	for (i = 0; i < ns; i++)
		rank[i] = sem_rank(sels[i], sem_c_eff(costs[i], hits[i], cache_cost));
	sem_order(rank, ns, ord);
	for (i = 0; i < ns; i++)
		out[i] = Int32GetDatum(ord[i]);

	PG_RETURN_ARRAYTYPE_P(construct_array(out, ns, INT4OID, sizeof(int32), true, TYPALIGN_INT));
}

/*
 * tridb_sem_plan_cost(sels, costs, hits, ord int[], n_in float8, cache_cost float8 DEFAULT 0.0)
 *   RETURNS float8 — n_in * sum_pos c_eff[ord[pos]] * prod_{q<pos} sels[ord[q]].
 */
PG_FUNCTION_INFO_V1(tridb_sem_plan_cost_sql);
Datum
tridb_sem_plan_cost_sql(PG_FUNCTION_ARGS)
{
	const char *fname = "tridb_sem_plan_cost";
	int			ns,
				nc,
				nh,
				no,
				pos;
	double	   *sels = sem_get_float8_array(PG_GETARG_ARRAYTYPE_P(0), fname, "sels", &ns);
	double	   *costs = sem_get_float8_array(PG_GETARG_ARRAYTYPE_P(1), fname, "costs", &nc);
	double	   *hits = sem_get_float8_array(PG_GETARG_ARRAYTYPE_P(2), fname, "hits", &nh);
	int32	   *ord = sem_get_int4_array(PG_GETARG_ARRAYTYPE_P(3), fname, "ord", &no);
	double		n_in = PG_GETARG_FLOAT8(4);
	double		cache_cost = PG_GETARG_FLOAT8(5);
	bool		seen[TRIDB_SEM_MAX_PREDS];
	double		acc = 0.0;
	double		surv = 1.0;

	sem_validate(fname, sels, ns, costs, nc, hits, nh, cache_cost);
	if (!(n_in >= 0.0) || isinf(n_in))
		sem_invalid(fname, "n_in must be finite and >= 0");
	if (no != ns)
		sem_invalid(fname, "ord must be a permutation of 0..n-1");
	memset(seen, 0, sizeof(seen));
	for (pos = 0; pos < no; pos++)
	{
		if (ord[pos] < 0 || ord[pos] >= ns || seen[ord[pos]])
			sem_invalid(fname, "ord must be a permutation of 0..n-1");
		seen[ord[pos]] = true;
	}

	for (pos = 0; pos < no; pos++)
	{
		int32		j = ord[pos];
		double		step = sem_c_eff(costs[j], hits[j], cache_cost) * surv;

		acc = acc + step;
		surv = surv * sels[j];
	}

	PG_RETURN_FLOAT8(n_in * acc);
}

/*
 * tridb_sem_batch(k_remaining int, sel_hat float8, min_batch int, max_batch int) RETURNS int
 *   k_remaining <= 0 -> 0; else clamp(ceil(k_remaining / max(sel_hat, 1e-6)), min_batch, max_batch).
 * The clamp is done in float8 BEFORE the int cast, so a huge quotient cannot overflow int32.
 */
PG_FUNCTION_INFO_V1(tridb_sem_batch_sql);
Datum
tridb_sem_batch_sql(PG_FUNCTION_ARGS)
{
	const char *fname = "tridb_sem_batch";
	int32		k = PG_GETARG_INT32(0);
	double		sel_hat = PG_GETARG_FLOAT8(1);
	int32		min_batch = PG_GETARG_INT32(2);
	int32		max_batch = PG_GETARG_INT32(3);
	double		s,
				q,
				v;

	if (!(sel_hat >= 0.0 && sel_hat <= 1.0))
		sem_invalid(fname, "sel_hat must be in [0, 1]");
	if (min_batch < 1 || max_batch < min_batch)
		sem_invalid(fname, "require 1 <= min_batch <= max_batch");
	if (k <= 0)
		PG_RETURN_INT32(0);

	s = (sel_hat > TRIDB_SEM_SEL_FLOOR) ? sel_hat : TRIDB_SEM_SEL_FLOOR;
	q = (double) k / s;
	v = ceil(q);
	if (v < (double) min_batch)
		PG_RETURN_INT32(min_batch);
	if (v > (double) max_batch)
		PG_RETURN_INT32(max_batch);
	PG_RETURN_INT32((int32) v);
}
