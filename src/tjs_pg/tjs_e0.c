/*
 * tjs_e0.c -- benchmark-internal E0 executor.
 *
 * Included by tjs_pg.c after the production tjs_open implementation.  It deliberately
 * does not alter graph_store.graph_query() or advertise a second query language.  The
 * SQL declarations live under experiments/e0 and are installed only in benchmark DBs.
 */

static bool
tjs_e0_filter_pass(SPIPlanPtr plan, int64 id)
{
	Datum		values[1] = {Int64GetDatum(id)};
	int			rc;

	if (plan == NULL)
		return true;
	rc = SPI_execute_plan(plan, values, NULL, true, 1);
	if (rc != SPI_OK_SELECT)
		ereport(ERROR, (errmsg("tjs_e0_open: relational predicate probe failed (%d)", rc)));
	return SPI_processed == 1;
}

static SPIPlanPtr
tjs_e0_prepare_filter(const char *nspname, const char *relname,
					  const char *id_col, const char *filter)
{
	StringInfoData sql;
	Oid			types[1] = {INT8OID};
	SPIPlanPtr	plan;

	if (filter[0] == '\0')
		return NULL;
	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT 1 FROM %s.%s WHERE %s = $1 AND (%s)",
					 qident(nspname), qident(relname), qident(id_col), filter);
	plan = SPI_prepare(sql.data, 1, types);
	if (plan == NULL)
		ereport(ERROR, (errmsg("tjs_e0_open: filter plan failed: %s", sql.data)));
	return plan;
}

static Portal
tjs_e0_graph_portal(Datum anchors, int32 hops, Datum type_ids, bool require_all)
{
	Oid			types[5] = {INT8ARRAYOID, INT4OID, INT4ARRAYOID, BOOLOID, INT8OID};
	Datum		values[5];

	values[0] = anchors;
	values[1] = Int32GetDatum(hops);
	values[2] = type_ids;
	values[3] = BoolGetDatum(require_all);
	values[4] = Int64GetDatum((int64) tjs_graph_work_budget);
	return SPI_cursor_open_with_args(NULL,
		"SELECT graph_store.gph_traverse_bounded_multi($1,$2,$3,$4,$5)",
		5, types, values, NULL, true, 0);
}

static bool
tjs_e0_fetch_graph(Portal portal, int64 *vid)
{
	bool		isnull;
	Datum		value;

	SPI_cursor_fetch(portal, true, 1);
	if (SPI_processed == 0)
		return false;
	value = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
	if (isnull)
		return tjs_e0_fetch_graph(portal, vid);
	*vid = DatumGetInt64(value);
	return true;
}

PG_FUNCTION_INFO_V1(tjs_e0_open_pg);
Datum
tjs_e0_open_pg(PG_FUNCTION_ARGS)
{
	Oid			reloid = PG_GETARG_OID(0);
	int32		candidate_k = PG_GETARG_INT32(1);
	int32		output_k = PG_GETARG_INT32(2);
	int32		hops = PG_GETARG_INT32(3);
	char	   *id_col;
	char	   *filter;
	Datum		query_vec = PG_GETARG_DATUM(6);
	Datum		anchors = PG_GETARG_DATUM(7);
	Datum		type_ids = PG_GETARG_DATUM(8);
	bool		require_all = PG_GETARG_BOOL(9);
	char	   *shape;
	char	   *placement;
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	Tuplestorestate *tupstore;
	TupleDesc	tupdesc;
	MemoryContext oldctx;
	Relation	heap;
	AttrNumber	vec_attno = InvalidAttrNumber;
	Oid			distproc = InvalidOid;
	Oid			distop = InvalidOid;
	FmgrInfo	distfn;
	char	   *vec_col;
	char	   *relname;
	char	   *nspname;
	char	   *opname;
	Oid			query_type;

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3) ||
		PG_ARGISNULL(4) || PG_ARGISNULL(5) || PG_ARGISNULL(6) || PG_ARGISNULL(7) ||
		PG_ARGISNULL(8) || PG_ARGISNULL(9) || PG_ARGISNULL(10) || PG_ARGISNULL(11))
		ereport(ERROR, (errmsg("tjs_e0_open: all arguments must be non-NULL")));
	if (candidate_k <= 0 || candidate_k > 10000 || output_k <= 0 || output_k > 10000)
		ereport(ERROR, (errmsg("tjs_e0_open: candidate_k/output_k must be in 1..10000")));
	if (hops < 0 || hops > 8)
		ereport(ERROR, (errmsg("tjs_e0_open: hops must be in 0..8")));

	id_col = text_to_cstring(PG_GETARG_TEXT_PP(4));
	filter = text_to_cstring(PG_GETARG_TEXT_PP(5));
	shape = text_to_cstring(PG_GETARG_TEXT_PP(10));
	placement = text_to_cstring(PG_GETARG_TEXT_PP(11));
	if (strcmp(shape, "vector_first") != 0 && strcmp(shape, "filter_first") != 0 &&
		strcmp(shape, "traverse_first") != 0)
		ereport(ERROR, (errmsg("tjs_e0_open: unknown shape \"%s\"", shape)));
	if (strcmp(placement, "pre") != 0 && strcmp(placement, "during") != 0 &&
		strcmp(placement, "post") != 0)
		ereport(ERROR, (errmsg("tjs_e0_open: unknown predicate placement \"%s\"", placement)));
	if ((strcmp(shape, "filter_first") == 0 && strcmp(placement, "pre") != 0) ||
		(strcmp(shape, "filter_first") != 0 && strcmp(placement, "pre") == 0))
		ereport(ERROR, (errmsg("tjs_e0_open: invalid shape/placement combination")));

	if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
		ereport(ERROR, (errmsg("tjs_e0_open: set-valued function called in non-set context")));
	oldctx = MemoryContextSwitchTo(rsinfo->econtext->ecxt_per_query_memory);
	tupdesc = CreateTemplateTupleDesc(1);
	TupleDescInitEntry(tupdesc, (AttrNumber) 1, "id", INT8OID, -1, 0);
	tupstore = tuplestore_begin_heap(false, false, work_mem);
	rsinfo->returnMode = SFRM_Materialize;
	rsinfo->setResult = tupstore;
	rsinfo->setDesc = tupdesc;
	MemoryContextSwitchTo(oldctx);

	tjs_examined = 0;
	tjs_graph_examined = 0;
	tjs_graph_censored = false;
	tjs_e0_vector_us = 0;
	tjs_e0_graph_us = 0;
	tjs_e0_filter_us = 0;
	tjs_e0_termination = "not_run";
	tjs_term_reason = TJS_TERM_FILTER_FIRST;

	heap = table_open(reloid, AccessShareLock);
	(void) find_hnsw_index(heap, &vec_attno, &distproc, &distop);
	vec_col = pstrdup(get_attname(reloid, vec_attno, false));
	relname = pstrdup(get_rel_name(reloid));
	nspname = pstrdup(get_namespace_name(get_rel_namespace(reloid)));
	opname = get_opname(distop);
	if (opname == NULL ||
		(strcmp(opname, "<->") != 0 && strcmp(opname, "<#>") != 0 &&
		 strcmp(opname, "<=>") != 0 && strcmp(opname, "<+>") != 0))
		ereport(ERROR, (errmsg("tjs_e0_open: unsupported HNSW distance operator")));
	fmgr_info(distproc, &distfn);
	table_close(heap, AccessShareLock);
	query_type = get_fn_expr_argtype(fcinfo->flinfo, 6);
	if (!OidIsValid(query_type))
		ereport(ERROR, (errmsg("tjs_e0_open: cannot resolve query vector type")));

	if (SPI_connect() != SPI_OK_CONNECT)
		ereport(ERROR, (errmsg("tjs_e0_open: SPI_connect failed")));

	if (strcmp(shape, "traverse_first") != 0)
	{
		StringInfoData sql;
		Oid			argtypes[1] = {query_type};
		Datum		values[1] = {query_vec};
		TopkItem   *candidates;
		bool	   *passes;
		bool	   *matched;
		int			ncand;
		int			pending = 0;
		int			i;
		int			remaining_out = output_k;
		SPIPlanPtr	filter_plan;
		TimestampTz stage;

		initStringInfo(&sql);
		appendStringInfo(&sql, "SELECT %s, %s %s $1 AS dist FROM %s.%s",
						 qident(id_col), qident(vec_col), opname,
						 qident(nspname), qident(relname));
		if (strcmp(shape, "filter_first") == 0 && filter[0] != '\0')
			appendStringInfo(&sql, " WHERE (%s)", filter);
		appendStringInfo(&sql, " ORDER BY %s %s $1 LIMIT %d",
						 qident(vec_col), opname, candidate_k);
		stage = GetCurrentTimestamp();
		if (SPI_execute_with_args(sql.data, 1, argtypes, values, NULL, true, 0) != SPI_OK_SELECT)
			ereport(ERROR, (errmsg("tjs_e0_open: ANN seed query failed: %s", sql.data)));
		tjs_e0_vector_us += GetCurrentTimestamp() - stage;
		ncand = (int) SPI_processed;
		candidates = (TopkItem *) palloc0(sizeof(TopkItem) * Max(ncand, 1));
		passes = (bool *) palloc0(sizeof(bool) * Max(ncand, 1));
		matched = (bool *) palloc0(sizeof(bool) * Max(ncand, 1));
		for (i = 0; i < ncand; i++)
		{
			bool idnull;
			bool dnull;

			candidates[i].id = DatumGetInt64(SPI_getbinval(
				SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1, &idnull));
			candidates[i].dist = DatumGetFloat8(SPI_getbinval(
				SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 2, &dnull));
			if (idnull || dnull)
				ereport(ERROR, (errmsg("tjs_e0_open: ANN seed returned NULL id/distance")));
			passes[i] = true;
		}
		tjs_examined = ncand;
		filter_plan = tjs_e0_prepare_filter(nspname, relname, id_col, filter);
		for (i = 0; i < ncand; i++)
		{
			if (strcmp(shape, "vector_first") == 0 && strcmp(placement, "during") == 0)
			{
				stage = GetCurrentTimestamp();
				passes[i] = tjs_e0_filter_pass(filter_plan, candidates[i].id);
				tjs_e0_filter_us += GetCurrentTimestamp() - stage;
			}
			if (passes[i])
				pending++;
		}

		if (pending > 0)
		{
			Portal portal;
			int64 v0 = tjs_read_graph_visits();
			int64 reached;

			stage = GetCurrentTimestamp();
			portal = tjs_e0_graph_portal(anchors, hops, type_ids, require_all);
			while (tjs_e0_fetch_graph(portal, &reached))
			{
				for (i = 0; i < ncand; i++)
					if (passes[i] && !matched[i] && candidates[i].id == reached)
					{
						matched[i] = true;
						pending--;
					}
				if (pending == 0)
				{
					tjs_e0_termination = "all_candidates_decided";
					break;
				}
			}
			SPI_cursor_close(portal);
			tjs_e0_graph_us += GetCurrentTimestamp() - stage;
			tjs_graph_examined = tjs_read_graph_visits() - v0;
			tjs_graph_censored = tjs_read_graph_censored();
			if (tjs_graph_censored)
				tjs_e0_termination = "graph_budget";
			else if (strcmp(tjs_e0_termination, "not_run") == 0)
				tjs_e0_termination = "graph_exhausted";
		}
		else
			tjs_e0_termination = "predicate_empty";

		for (i = 0; i < ncand && remaining_out > 0; i++)
		{
			bool emit = matched[i] && passes[i];

			if (emit && strcmp(shape, "vector_first") == 0 && strcmp(placement, "post") == 0)
			{
				stage = GetCurrentTimestamp();
				emit = tjs_e0_filter_pass(filter_plan, candidates[i].id);
				tjs_e0_filter_us += GetCurrentTimestamp() - stage;
			}
			if (emit)
			{
				Datum row[1] = {Int64GetDatum(candidates[i].id)};
				bool nulls[1] = {false};

				tuplestore_putvalues(tupstore, tupdesc, row, nulls);
				remaining_out--;
			}
		}
	}
	else
	{
		StringInfoData fq;
		Oid			ftypes[1] = {INT8OID};
		SPIPlanPtr	fetch_plan;
		Portal		portal;
		Topk		topk;
		int64		v0;
		int64		reached;
		int			i;
		TimestampTz graph_start;

		initStringInfo(&fq);
		appendStringInfo(&fq, "SELECT %s FROM %s.%s WHERE %s = $1",
						 qident(vec_col), qident(nspname), qident(relname), qident(id_col));
		if (filter[0] != '\0')
			appendStringInfo(&fq, " AND (%s)", filter);
		fetch_plan = SPI_prepare(fq.data, 1, ftypes);
		if (fetch_plan == NULL)
			ereport(ERROR, (errmsg("tjs_e0_open: traverse fetch plan failed")));
		topk_init(&topk, candidate_k);
		v0 = tjs_read_graph_visits();
		graph_start = GetCurrentTimestamp();
		portal = tjs_e0_graph_portal(anchors, hops, type_ids, require_all);
		while (tjs_e0_fetch_graph(portal, &reached))
		{
			Datum ids[1] = {Int64GetDatum(reached)};
			TimestampTz filter_start = GetCurrentTimestamp();
			int rc = SPI_execute_plan(fetch_plan, ids, NULL, true, 1);

			tjs_e0_filter_us += GetCurrentTimestamp() - filter_start;
			if (rc != SPI_OK_SELECT)
				ereport(ERROR, (errmsg("tjs_e0_open: traverse row fetch failed")));
			if (SPI_processed == 1)
			{
				bool vnull;
				Datum vd = SPI_getbinval(SPI_tuptable->vals[0],
									  SPI_tuptable->tupdesc, 1, &vnull);

				if (!vnull)
				{
					TimestampTz vector_start = GetCurrentTimestamp();
					float8 dist = DatumGetFloat8(
						FunctionCall2Coll(&distfn, InvalidOid, vd, query_vec));

					tjs_e0_vector_us += GetCurrentTimestamp() - vector_start;
					(void) topk_offer(&topk, reached, dist);
					tjs_examined++;
				}
			}
		}
		SPI_cursor_close(portal);
		tjs_e0_graph_us = GetCurrentTimestamp() - graph_start -
			tjs_e0_filter_us - tjs_e0_vector_us;
		if (tjs_e0_graph_us < 0)
			tjs_e0_graph_us = 0;
		tjs_graph_examined = tjs_read_graph_visits() - v0;
		tjs_graph_censored = tjs_read_graph_censored();
		tjs_e0_termination = tjs_graph_censored ? "graph_budget" : "graph_exhausted";
		qsort(topk.items, topk.n, sizeof(TopkItem), topk_cmp_dist);
		for (i = 0; i < topk.n && i < output_k; i++)
		{
			Datum row[1] = {Int64GetDatum(topk.items[i].id)};
			bool nulls[1] = {false};

			tuplestore_putvalues(tupstore, tupdesc, row, nulls);
		}
	}

	SPI_finish();
	pfree(id_col);
	pfree(filter);
	pfree(shape);
	pfree(placement);
	return (Datum) 0;
}

PG_FUNCTION_INFO_V1(tjs_e0_vector_us_pg);
Datum
tjs_e0_vector_us_pg(PG_FUNCTION_ARGS)
{
	PG_RETURN_INT64(tjs_e0_vector_us);
}

PG_FUNCTION_INFO_V1(tjs_e0_graph_us_pg);
Datum
tjs_e0_graph_us_pg(PG_FUNCTION_ARGS)
{
	PG_RETURN_INT64(tjs_e0_graph_us);
}

PG_FUNCTION_INFO_V1(tjs_e0_filter_us_pg);
Datum
tjs_e0_filter_us_pg(PG_FUNCTION_ARGS)
{
	PG_RETURN_INT64(tjs_e0_filter_us);
}

PG_FUNCTION_INFO_V1(tjs_e0_termination_pg);
Datum
tjs_e0_termination_pg(PG_FUNCTION_ARGS)
{
	PG_RETURN_TEXT_P(cstring_to_text(tjs_e0_termination));
}
