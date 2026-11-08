"""Статистика A/B-тестов на агрегатах групп.

Все функции принимают суммы по группам, а не сырые массивы: так одну и ту же формулу можно
применить сразу к тысячам случайных разбиений (A/A-тесты) через матричное умножение.
"""
import numpy as np
from scipy import stats

Z_ALPHA = stats.norm.ppf(0.975)
Z_POWER = stats.norm.ppf(0.80)


def group_sums(cols, G):
    """Суммы по тестовой группе для каждого разбиения и по контрольной как остаток.

    cols — словарь {имя: вектор по пользователям}, G — матрица n × S из 0/1 (1 — тест).
    """
    t = {k: v @ G for k, v in cols.items()}
    t['n'] = G.sum(axis=0)
    c = {k: v.sum() - t[k] for k, v in cols.items()}
    c['n'] = G.shape[0] - t['n']
    return t, c


def welch_p(n1, s1, q1, n0, s0, q0):
    """t-тест Уэлча по суммам и суммам квадратов. Возвращает (разница средних, p-value)."""
    m1, m0 = s1 / n1, s0 / n0
    v1 = (q1 - n1 * m1 ** 2) / (n1 - 1)
    v0 = (q0 - n0 * m0 ** 2) / (n0 - 1)
    se = np.sqrt(v1 / n1 + v0 / n0)
    z = (m1 - m0) / se
    return m1 - m0, 2 * stats.norm.sf(np.abs(z))


def prop_p(n1, k1, n0, k0):
    """z-тест для двух долей с объединённой дисперсией."""
    p = (k1 + k0) / (n1 + n0)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n0))
    z = (k1 / n1 - k0 / n0) / se
    return k1 / n1 - k0 / n0, 2 * stats.norm.sf(np.abs(z))


def ratio_var(n, sx, sy, qx, qy, sxy):
    """Дисперсия отношения ΣX/ΣY по дельта-методу (пользователь — единица рандомизации)."""
    mx, my = sx / n, sy / n
    vx = qx / n - mx ** 2
    vy = qy / n - my ** 2
    cxy = sxy / n - mx * my
    return (vx / my ** 2 - 2 * mx * cxy / my ** 3 + mx ** 2 * vy / my ** 4) / n


def delta_p(t, c, x, y):
    """Сравнение метрик-отношений дельта-методом. t, c — суммы групп, x/y — имена числителя и знаменателя."""
    r1, r0 = t[x] / t[y], c[x] / c[y]
    v1 = ratio_var(t['n'], t[x], t[y], t[x + '2'], t[y + '2'], t[x + y])
    v0 = ratio_var(c['n'], c[x], c[y], c[x + '2'], c[y + '2'], c[x + y])
    z = (r1 - r0) / np.sqrt(v1 + v0)
    return r1 - r0, 2 * stats.norm.sf(np.abs(z))


def naive_ratio_p(t, c, x, y, xsq_unit):
    """Наивный тест отношения: каждое событие знаменателя считается независимым наблюдением.

    Для бинарного числителя (клик на показ) это z-тест долей по показам, для счётного
    (товары на день покупки) — t-тест по дням. xsq_unit — имя суммы квадратов числителя
    по единицам знаменателя или None для бинарного случая.
    """
    if xsq_unit is None:
        return prop_p(t[y], t[x], c[y], c[x])
    return welch_p(t[y], t[x], t[xsq_unit], c[y], c[x], c[xsq_unit])


def cuped(y, x):
    """CUPED: y − θ·(x − x̄), θ = cov(y, x) / var(x). Возвращает новый вектор и долю снижения дисперсии."""
    theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    yc = y - theta * (x - x.mean())
    return yc, 1 - np.var(yc) / np.var(y)


def mde_rel(var1, var0, n1, n0, base):
    """Минимальный обнаруживаемый эффект при α = 5% и мощности 80%, в долях от базового значения."""
    return (Z_ALPHA + Z_POWER) * np.sqrt(var1 / n1 + var0 / n0) / base


def srm_p(n1, n0, share=0.5):
    """χ²-проверка соответствия размеров групп плановому разбиению (sample ratio mismatch)."""
    n = n1 + n0
    return stats.chisquare([n1, n0], [n * share, n * (1 - share)]).pvalue


def holm(pvals, alpha=0.05):
    """Поправка Холма: какие гипотезы отвергаются при контроле FWER."""
    p = np.asarray(pvals)
    order = np.argsort(p)
    reject = np.zeros(len(p), bool)
    for rank, i in enumerate(order):
        if p[i] > alpha / (len(p) - rank):
            break
        reject[i] = True
    return reject
