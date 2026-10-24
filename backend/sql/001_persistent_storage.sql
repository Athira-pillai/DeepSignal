-- Run once in the Supabase Dashboard's SQL Editor. No local database install is needed.
create extension if not exists pgcrypto with schema extensions;

create table if not exists public.market_data_runs (
    id uuid primary key default gen_random_uuid(),
    symbol text not null,
    interval text not null,
    source text not null,
    market_timestamp timestamptz not null,
    retrieved_at timestamptz not null,
    data_sha256 text not null,
    row_count integer not null check (row_count >= 0),
    quality_status text not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (symbol, interval, data_sha256)
);

create table if not exists public.predictions (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    symbol text not null,
    horizon integer not null check (horizon > 0),
    prediction_timestamp timestamptz not null,
    signal_type text not null,
    predicted_trend text not null,
    predicted_change numeric,
    ensemble_probability_up numeric check (ensemble_probability_up between 0 and 1),
    model_version text not null,
    input_window_start date,
    input_window_end date,
    input_features jsonb not null default '{}'::jsonb,
    market_data_run_id uuid references public.market_data_runs(id),
    response jsonb not null
);

create table if not exists public.signals (
    id uuid primary key default gen_random_uuid(),
    prediction_id uuid not null unique references public.predictions(id) on delete cascade,
    symbol text not null,
    signal_type text not null,
    strategy_version text not null,
    signal_timestamp timestamptz not null,
    reason text,
    risk_control_decision text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists public.experiments (
    id uuid primary key default gen_random_uuid(),
    run_id text not null unique,
    experiment_type text not null check (experiment_type in ('evaluation', 'backtest')),
    symbol text not null,
    created_at timestamptz not null default now(),
    model_version text,
    strategy_version text,
    window_start date,
    window_end date,
    data_sha256 text,
    periods jsonb not null default '{}'::jsonb,
    metrics jsonb not null default '{}'::jsonb,
    parameters jsonb not null default '{}'::jsonb,
    assumptions jsonb not null default '{}'::jsonb,
    report jsonb not null
);

create index if not exists predictions_symbol_time_idx
    on public.predictions (symbol, prediction_timestamp desc);
create index if not exists signals_symbol_time_idx
    on public.signals (symbol, signal_timestamp desc);
create index if not exists experiments_type_time_idx
    on public.experiments (experiment_type, created_at desc);

alter table public.market_data_runs enable row level security;
alter table public.predictions enable row level security;
alter table public.signals enable row level security;
alter table public.experiments enable row level security;

-- Only the backend's Supabase secret key can access these private tables.
revoke all on public.market_data_runs, public.predictions, public.signals, public.experiments from public, anon, authenticated;
grant usage on schema public to service_role;
grant select, insert, update on public.market_data_runs, public.predictions, public.signals, public.experiments to service_role;
grant usage, select on all sequences in schema public to service_role;
