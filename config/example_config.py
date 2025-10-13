SIMULATION_CONFIG = {
    'initial_capital': 100000,
    'years': 10,
    'simulations': 10000,
    'method': 'bootstrap'  # or 'geometric_brownian'
}

ASSETS = {
    'voo': {
        'ticker': 'VOO',
        'name': 'VOO (S&P 500)',
        'lookback_years': 10
    },
    'qqq': {
        'ticker': 'QQQ',
        'name': 'QQQ (Nasdaq)',
        'lookback_years': 10
    },
}

PORTFOLIOS = [
    {
        'name': 'My Portfolio',
        'description': 'Custom allocation',
        'allocations': {
            'voo': 0.6,
            'qqq': 0.4
        }
    }
]

OPTIMIZATION_CONFIG = {
    'enabled': True,
    'method': 'grid_search',
    'objective_weights': {
        'return': 0.50,
        'sharpe': 0.20,
        'drawdown': 0.30
    },
    'optimize_assets': ['voo', 'qqq']
}