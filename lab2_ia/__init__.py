import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

class BayesModel:

    '''
    Clase que representa el modelo bayesiano
    Recibe como parámetro un diccionario con los parámetros del modelo
        - ['nodos']
        - ['edge'] donde en la posicion 0 se encuentra el nodo padre y en la posicion 1 el nodo hijo
        - ['probabilidad'] donde en la posicion 0 se encuentra el nodo y en la posicion 1 un array de las probabilidades
    '''
    def __init__(self, parameters):
        self.parameters = parameters
        self.model = BayesianModel(parameters)
        self.inference = None
    

    '''
    Construye el modelo bayesiano a partir de los parámetros recibidos en el constructor
    '''
    def build_model(self):
        for nodos in self.parameters['nodos']:
            self.model.add_node(nodos)
        
        for edge in self.parameters['edge']:
            self.model.add_edge(edge[0], edge[1])

        
        for probabilidad in self.parameters['probabilidad']:
            temp_cpd = TabularCPD(probabilidad[0], 2,  values = [probabilidad[1]])
            self.model.add_cpd(temp_cpd)

    def get_inference(self):    
        return self.inference

    def get_model(self):
        return self.model

    def variable_elimination(self, varibles, evidence):
        self.inference = pgmpy.inference.VariableElimination(self.model)
        return self.inference.query(variables=varibles, evidence=evidence)
    
    def obtener_factores(self):
        return self.model.get_factors()

    def representacion_compacta(self):
        return self.model.to_string()

    def completamente_descrita(self):
        return self.model.check_model()

