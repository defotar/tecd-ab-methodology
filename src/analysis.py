"""Проверка методики A/B-тестов на исторических данных T-ECD.

1. A/A-тесты: доля ложных срабатываний правильных и наивных критериев.
2. Мощность: MDE за 28 дней и по длительности теста, без CUPED и с CUPED.
3. Подглядывание: как растут ложные победы, если смотреть результат каждый день.
4. Множественные сравнения и SRM.
5. Пример теста с внесённым эффектом и симуляция мощности.

Результаты — reports/results.json, графики — reports/figures.
"""
import json

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import abtools as ab

RNG = np.random.default_rng(20261108)
N_AA = 2000
N_PEEK = 1000
N_POWER = 1000
POCOCK_4 = 0.0182  # номинальный порог Покока для 4 промежуточных проверок при α = 5%
FIG = 'reports/figures/'
plt.rcParams.update({'figure.dpi': 130, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
BLUE, RED, GREY = '#2b6cb0', '#c53030', '#a0aec0'

res = {}

# --- Данные ---------------------------------------------------------------------------------------
ru = pd.read_parquet('data/processed/retail_users.parquet')
mu = pd.read_parquet('data/processed/market_users.parquet')
daily = pd.read_parquet('data/processed/retail_daily.parquet')

# Сумма квадратов товаров по дням покупки — нужна только наивному тесту «товаров в дне покупки»
sq = daily[daily.buy_day == 1].assign(i2=lambda d: d['items'].astype(float) ** 2).groupby('user_id').i2.sum()
ru['items_sq_days'] = ru.user_id.map(sq).fillna(0)

f = lambda s: s.to_numpy(float)
R = {
    'buy': f(ru['items'] > 0), 'items': f(ru['items']), 'buy_days': f(ru.buy_days),
    'items_sq_days': f(ru.items_sq_days),
}
R['items2'], R['buy_days2'], R['itemsbuy_days'] = R['items'] ** 2, R['buy_days'] ** 2, R['items'] * R['buy_days']
R['items_sq'] = R['items'] ** 2

M = {'clicks': f(mu.clicks), 'views': f(mu.views), 'out': f(mu.clickouts > 0)}
M['clicks2'], M['views2'], M['clicksviews'] = M['clicks'] ** 2, M['views'] ** 2, M['clicks'] * M['views']

res['data'] = {
    'retail_users': len(ru), 'market_users': len(mu),
    'retail_buy_rate': R['buy'].mean(), 'retail_items_per_user': R['items'].mean(),
    'retail_items_per_buy_day': R['items'].sum() / R['buy_days'].sum(),
    'market_ctr': M['clicks'].sum() / M['views'].sum(), 'market_clickout_rate': M['out'].mean(),
    'retail_items_p99': float(np.percentile(R['items'], 99)),
}

METRICS = {
    'buy': 'Конверсия в покупку (ритейл)',
    'items': 'Товаров на пользователя (ритейл)',
    'basket': 'Товаров в дне покупки (ритейл)',
    'ctr': 'CTR (маркетплейс)',
    'out': 'Конверсия в переход к партнёру (маркетплейс)',
}


# --- 1. A/A-тесты ------------------------------------------------------------------------------------
def aa_retail(G):
    t, c = ab.group_sums(R, G)
    return {
        'buy': ab.prop_p(t['n'], t['buy'], c['n'], c['buy'])[1],
        'items': ab.welch_p(t['n'], t['items'], t['items_sq'], c['n'], c['items'], c['items_sq'])[1],
        'basket': ab.delta_p(t, c, 'items', 'buy_days')[1],
        'basket_naive': ab.naive_ratio_p(t, c, 'items', 'buy_days', 'items_sq_days')[1],
    }


def aa_market(G):
    t, c = ab.group_sums(M, G)
    return {
        'ctr': ab.delta_p(t, c, 'clicks', 'views')[1],
        'ctr_naive': ab.naive_ratio_p(t, c, 'clicks', 'views', None)[1],
        'out': ab.prop_p(t['n'], t['out'], c['n'], c['out'])[1],
    }


def run_aa(fn, n, splits, chunk):
    out = {}
    for i in range(0, splits, chunk):
        G = (RNG.random((n, min(chunk, splits - i))) < 0.5).astype(float)
        for k, v in fn(G).items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


p = run_aa(aa_retail, len(ru), N_AA, 500) | run_aa(aa_market, len(mu), N_AA, 40)
fpr = {k: float((v < 0.05).mean()) for k, v in p.items()}
correct = ['buy', 'items', 'basket', 'ctr', 'out']
P5 = np.column_stack([p[k] for k in correct])
res['aa'] = {
    'splits': N_AA, 'fpr': fpr,
    'fwer_uncorrected': float((P5 < 0.05).any(axis=1).mean()),
    'fwer_holm': float(np.mean([ab.holm(row).any() for row in P5])),
    'fwer_bonferroni': float((P5 < 0.05 / 5).any(axis=1).mean()),
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
labels = ['Конверсия\nв покупку', 'Товаров\nна польз.', 'Товаров в дне\nпокупки', 'CTR', 'Конверсия\nв переход']
naive = [None, None, fpr['basket_naive'], fpr['ctr_naive'], None]
x = np.arange(5)
axes[0].bar(x - 0.2, [fpr[k] * 100 for k in correct], 0.4, color=BLUE, label='правильный критерий')
axes[0].bar(x + 0.2, [v * 100 if v else 0 for v in naive], 0.4, color=RED, label='наивный: событие = наблюдение')
axes[0].axhline(5, color='black', lw=1, ls='--')
axes[0].text(4.6, 5.6, 'α = 5%', ha='right', fontsize=9)
for i, v in enumerate(naive):
    if v:
        axes[0].text(i + 0.2, v * 100 + 0.8, f'{v * 100:.0f}%', ha='center', color=RED, fontsize=9)
axes[0].set_xticks(x, labels, fontsize=8)
axes[0].set_ylabel('Ложные срабатывания, %')
axes[0].set_title(f'A/A-тесты: {N_AA} случайных разбиений')
axes[0].legend(fontsize=8, frameon=False)
bins = np.linspace(0, 1, 21)
axes[1].hist(p['ctr_naive'], bins, color=RED, alpha=0.6, label='CTR, наивный z-тест по показам')
axes[1].hist(p['ctr'], bins, color=BLUE, alpha=0.7, label='CTR, дельта-метод')
axes[1].axhline(N_AA / 20, color='black', lw=1, ls='--')
axes[1].set_xlabel('p-value')
axes[1].set_ylabel('Число разбиений')
axes[1].set_title('Распределение p-value в A/A-тестах\n(у верного критерия — равномерное)')
axes[1].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(FIG + '01_aa_tests.png')
plt.close(fig)


# --- 2. CUPED и мощность -----------------------------------------------------------------------------
def linearize(x, y, xp, yp):
    """Линеаризация отношения: L = X − R·Y, где R — общее значение метрики. Для предпериода — то же с его R."""
    r, rp = x.sum() / y.sum(), xp.sum() / max(yp.sum(), 1)
    return (x - r * y) / y.mean(), (xp - rp * yp) / y.mean()


pairs = {
    'buy': (R['buy'], f(ru.items_pre > 0)),
    'items': (R['items'], f(ru.items_pre)),
    'basket': linearize(R['items'], R['buy_days'], f(ru.items_pre), f(ru.buy_days_pre)),
    'ctr': linearize(M['clicks'], M['views'], f(mu.clicks_pre), f(mu.views_pre)),
    'out': (M['out'], f(mu.clickouts_pre > 0)),
}
base = {'buy': R['buy'].mean(), 'items': R['items'].mean(), 'basket': 1.0, 'ctr': 1.0, 'out': M['out'].mean()}
# Линеаризованная метрика отнормирована на среднее знаменателя, поэтому её MDE считается от самого отношения
ratio_base = {'basket': res['data']['retail_items_per_buy_day'], 'ctr': res['data']['market_ctr']}

mde = {}
for k, (y, xpre) in pairs.items():
    n = len(y) / 2
    yc, red = ab.cuped(y, xpre)
    b = ratio_base.get(k, base[k])
    mde[k] = {
        'var_reduction': float(red),
        'mde_plain': float(ab.mde_rel(y.var(), y.var(), n, n, b)),
        'mde_cuped': float(ab.mde_rel(yc.var(), yc.var(), n, n, b)),
    }
res['mde_28d'] = mde

# MDE «товаров на пользователя» по длительности теста: пользователи, пришедшие к дню D, и их товары к дню D
days = np.arange(1281, 1309)
uid = {u: i for i, u in enumerate(ru.user_id)}
rows = daily.user_id.map(uid).to_numpy()
col = (daily.d - 1281).to_numpy()
item_day = np.zeros((len(ru), 28))
np.add.at(item_day, (rows, col), daily['items'].to_numpy(float))
cum = item_day.cumsum(axis=1)
seen = (ru.first_day.to_numpy()[:, None] <= days[None, :]).astype(float)
pre_items = f(ru.items_pre)

by_days = []
for D in (7, 14, 21, 28):
    m = seen[:, D - 1] == 1
    y, xp = cum[m, D - 1], pre_items[m]
    yc, red = ab.cuped(y, xp)
    n = m.sum() / 2
    by_days.append({'days': D, 'users': int(m.sum()),
                    'mde_plain': float(ab.mde_rel(y.var(), y.var(), n, n, y.mean())),
                    'mde_cuped': float(ab.mde_rel(yc.var(), yc.var(), n, n, y.mean())),
                    'var_reduction': float(red)})
res['mde_by_days_items'] = by_days

fig, ax = plt.subplots(figsize=(6.5, 4))
d_ = [r['days'] for r in by_days]
ax.plot(d_, [r['mde_plain'] * 100 for r in by_days], 'o-', color=GREY, label='обычный t-тест')
ax.plot(d_, [r['mde_cuped'] * 100 for r in by_days], 'o-', color=BLUE, label='CUPED (товары за 28 дней до теста)')
for r in by_days:
    ax.annotate(f"{r['mde_plain'] * 100:.1f}%", (r['days'], r['mde_plain'] * 100), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8, color='#4a5568')
    ax.annotate(f"{r['mde_cuped'] * 100:.1f}%", (r['days'], r['mde_cuped'] * 100), textcoords='offset points', xytext=(0, -14), ha='center', fontsize=8, color=BLUE)
ax.set_xticks(d_)
ax.set_xlabel('Длительность теста, дней')
ax.set_ylabel('MDE, % от среднего')
ax.set_ylim(bottom=0)
ax.set_title('Какой рост «товаров на пользователя» тест заметит\n(α = 5%, мощность 80%, разбиение 50/50)')
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(FIG + '02_mde_by_days.png')
plt.close(fig)

# --- 3. Подглядывание --------------------------------------------------------------------------------
cum2 = cum ** 2
pk = []
for i in range(0, N_PEEK, 250):
    G = (RNG.random((len(ru), min(250, N_PEEK - i))) < 0.5).astype(float)
    n1, s1, q1 = seen.T @ G, cum.T @ G, cum2.T @ G
    n0 = seen.sum(0)[:, None] - n1
    s0 = cum.sum(0)[:, None] - s1
    q0 = cum2.sum(0)[:, None] - q1
    pk.append(ab.welch_p(n1, s1, q1, n0, s0, q0)[1])  # 28 дней × разбиения
pk = np.concatenate(pk, axis=1)
ever = np.maximum.accumulate(pk < 0.05, axis=0).mean(axis=1)  # доля тестов, где к этому дню уже была «победа»
weekly = pk[[6, 13, 20, 27]]
res['peeking'] = {
    'splits': N_PEEK,
    'fpr_fixed_28d': float((pk[27] < 0.05).mean()),
    'fpr_daily_peeking': float((pk < 0.05).any(axis=0).mean()),
    'fpr_weekly_peeking_0_05': float((weekly < 0.05).any(axis=0).mean()),
    'fpr_weekly_pocock': float((weekly < POCOCK_4).any(axis=0).mean()),
}

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(np.arange(1, 29), ever * 100, color=RED, lw=2, label='смотрим каждый день, останавливаем при p < 0.05')
ax.axhline(res['peeking']['fpr_fixed_28d'] * 100, color=BLUE, lw=2, label='одна проверка в конце, 28 дней')
ax.axhline(5, color='black', lw=1, ls='--')
ax.text(28, ever[-1] * 100 + 0.8, f'{ever[-1] * 100:.0f}%', ha='right', color=RED)
ax.set_xlabel('День теста')
ax.set_ylabel('Доля A/A-тестов с «победой», %')
ax.set_ylim(0, max(ever) * 100 * 1.25)
ax.set_title('Подглядывание: ложные победы накапливаются')
ax.legend(frameon=False, fontsize=8, loc='upper left')
fig.tight_layout()
fig.savefig(FIG + '03_peeking.png')
plt.close(fig)

# --- 4. SRM: потеря событий части пользователей в тестовой группе ------------------------------------
G = RNG.random(len(ru)) < 0.5
u = RNG.random(len(ru))
res['srm_demo'] = []
for share in (0.05, 0.10, 0.20):
    lost = G & (R['buy'] == 0) & (u < share)  # у части неплативших в тесте не доходят события
    gt, gc = G & ~lost, ~G & ~lost
    k1, n1, k0, n0 = R['buy'][gt].sum(), gt.sum(), R['buy'][gc].sum(), gc.sum()
    diff, pval = ab.prop_p(n1, k1, n0, k0)
    res['srm_demo'].append({
        'lost_share_of_nonbuyers': share, 'lost_users': int(lost.sum()), 'lost_share_of_group': float(lost.sum() / G.sum()),
        'n_test': int(n1), 'n_control': int(n0), 'srm_p': float(ab.srm_p(n1, n0)),
        'buy_diff_pp': float(diff * 100), 'buy_p': float(pval),
    })

# --- 5. Пример теста: +X% к товарам на пользователя -------------------------------------------------
EFFECT = round(mde['items']['mde_cuped'] * 100) / 100  # эффект на уровне MDE с CUPED, округлённый до 1 п.п.
y0, xpre = R['items'], f(ru.items_pre)


def one_test(G, rng):
    y = y0 + G * rng.poisson(EFFECT * y0)
    out = {}
    for name, v in (('plain', y), ('cuped', ab.cuped(y, xpre)[0])):
        t, c = v[G == 1], v[G == 0]
        d = t.mean() - c.mean()
        se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
        base_c = y[G == 0].mean()
        out[name] = {'rel': d / base_c, 'lo': (d - ab.Z_ALPHA * se) / base_c, 'hi': (d + ab.Z_ALPHA * se) / base_c,
                     'p': float(2 * ab.stats.norm.sf(abs(d / se)))}
    return out, y


sims = [one_test((RNG.random(len(ru)) < 0.5).astype(float), RNG)[0] for _ in range(N_POWER)]
res['power_sim'] = {
    'effect': EFFECT, 'sims': N_POWER,
    'power_plain': float(np.mean([s['plain']['p'] < 0.05 for s in sims])),
    'power_cuped': float(np.mean([s['cuped']['p'] < 0.05 for s in sims])),
}

G = (np.random.default_rng(1).random(len(ru)) < 0.5).astype(float)
ex, y_ex = one_test(G, np.random.default_rng(2))
t_, c_ = G == 1, G == 0
guard = ab.prop_p(t_.sum(), R['buy'][t_].sum(), c_.sum(), R['buy'][c_].sum())
res['example'] = {
    'effect_injected': EFFECT, 'n_test': int(t_.sum()), 'n_control': int(c_.sum()),
    'srm_p': float(ab.srm_p(t_.sum(), c_.sum())), 'items': ex,
    'guardrail_buy_diff_pp': float(guard[0] * 100), 'guardrail_buy_p': float(guard[1]),
}

fig, ax = plt.subplots(figsize=(7.5, 3))
for i, (name, lab, colr) in enumerate((('plain', 'обычный t-тест', GREY), ('cuped', 'CUPED', BLUE))):
    e = ex[name]
    ax.errorbar(e['rel'] * 100, i, xerr=[[(e['rel'] - e['lo']) * 100], [(e['hi'] - e['rel']) * 100]],
                fmt='o', color=colr, capsize=5, lw=2)
    ax.text(e['hi'] * 100 + 0.4, i, f"{e['rel'] * 100:+.1f}% [{e['lo'] * 100:+.1f}; {e['hi'] * 100:+.1f}], p = {e['p']:.3f}",
            va='center', fontsize=8)
ax.axvline(0, color='black', lw=1)
ax.axvline(EFFECT * 100, color=RED, lw=1, ls='--')
ax.text(EFFECT * 100, 1.45, f'истинный эффект +{EFFECT * 100:.0f}%', color=RED, ha='center', fontsize=8)
ax.set_yticks([0, 1], ['обычный t-тест', 'CUPED'])
ax.set_ylim(-0.6, 1.7)
ax.set_xlim(-3, ex['plain']['hi'] * 100 + 10)
ax.set_xlabel('Эффект на «товаров на пользователя», % (95% ДИ)')
ax.set_title('Пример теста: один эксперимент, два способа анализа')
fig.tight_layout()
fig.savefig(FIG + '04_example_test.png')
plt.close(fig)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    return o


json.dump(clean(res), open('reports/results.json', 'w'), ensure_ascii=False, indent=2)
print(json.dumps(clean(res), ensure_ascii=False, indent=1))
