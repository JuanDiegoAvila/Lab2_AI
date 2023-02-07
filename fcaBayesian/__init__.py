from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
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
        self.model = BayesianModel()

    '''
    Construye el modelo bayesiano a partir de los parámetros recibidos en el constructor
    '''

    def build_model(self):
        for nodos in self.parameters['nodos']:
            self.model.add_node(nodos)

        for edge in self.parameters['edge']:
            self.model.add_edge(edge[0], edge[1])

        for probabilidad in self.parameters['probabilidad']:

            temp_cpd = TabularCPD(probabilidad[0], 2,  values=probabilidad[1],
                                  evidence=probabilidad[2], evidence_card=probabilidad[3])
            self.model.add_cpds(temp_cpd)
        print("\nModelo Bayesiano construído con éxito!\n")
        self.inference = VariableElimination(self.model)

    '''
    Método encargado de realizar una consulta a la base de conocimiento dada una serie de variables y evidencia
    '''

    def enumeracion(self, variables, evidence):
        en = self.inference.query(variables=variables, evidence=evidence)
        print("\nProbabilidad dada según las variables otorgadas:\n")
        return en

    '''
    Método encargado de devolver las dependencias independientes entre los nodos del modelo Bayesiano
    '''

    def obtener_factores(self):
        print("\nDependencias independientes entre los nodos del modelo Bayesiano:\n")
        return self.model.get_independencies()

    '''
    Método encargado de retornar una representación en string de las probabilidades condicionales del modelo Bayesiano
    '''

    def representacion_compacta(self):
        cpd_string = ""
        for cpd in self.model.get_cpds():
            cpd_string += str(cpd)

        print("\nRepresentación en string de las probabilidades condicionales del modelo Bayesiano:\n")
        return cpd_string

    '''
    Método encargado de devolver un resultado booleano indicando si el modelo Bayesiano está completamente descrito o no
    '''

    def completamente_descrita(self):
        print("\nResultado booleano si el modelo Bayesiano está completamente descrito o no:\n")
        return self.model.check_model()
