import fcaBayesian as fca

parametros = {
    'nodos': ['M', 'U', 'R', 'B', 'S'],
    'edge': [['M','R'], ['U','R'], ['B','R'], ['B','S'], ['R','S']],
    'probabilidad':     [['M', [[0.95], [0.05]],[], []], ['U', [[0.85], [0.15]],[], []], ['B', [[0.90], [0.10]],[], []], ['S', [[0.98, .88, .95, .6], [.02, .12, .05, .40]],['R','B'], [2,2]],
        ['R', [[0.96, .86, .94, .82, .24, .15, .10, .05], [.04, .14, .06, .18, .76, .85, .90, .95]],['M','B', 'U'], [2,2,2]]],

}

bayes = fca.BayesModel(parametros)

bayes.build_model()

print(bayes.representacion_compacta())

print(bayes.completamente_descrita())

print(bayes.obtener_factores())

print(bayes.enumeracion(['R'], {'M': 1,}))
