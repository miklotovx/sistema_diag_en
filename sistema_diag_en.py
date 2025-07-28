# Imports
import warnings                                                                      
warnings.filterwarnings("ignore")
import pandas as pd                                           
from sklearn.datasets import load_breast_cancer               
from sklearn.model_selection import train_test_split         
from sklearn.neural_network import MLPClassifier             
from sklearn.ensemble import RandomForestClassifier          
from sklearn.linear_model import LogisticRegression           


# Database class
class ClinicalDatabase:                                       
    def __init__(self):                                       
        dataset = load_breast_cancer(as_frame=True)           
        self.data = dataset.frame.drop(columns=["target"])    
        self.labels = dataset.frame["target"]                 
        self.feature_names = list(self.data.columns)          

    def get_features(self, feature_list):                     
        return self.data[feature_list]

    def get_labels(self):                                     
        return self.labels


# *** Model A - MLP ***

# Untrained model
class ModelA_MLP:                                                                                  
    def __init__(self, database: ClinicalDatabase):                                                
        self.model_type = "MLP"                                                                    
        self.database = database                                                                   
        self.trained_model = None                                                                  

    def build_model(self):                                                                         
        X = self.database.get_features(self.database.feature_names)                                
        y = self.database.get_labels()                                                             
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  
        model_a = MLPClassifier(max_iter=500, random_state=42)                                     
        model_a.fit(X_train, y_train)                                                              
        print("ModelA_MLP successfully trained.")
        self.trained_model = TrainedModelA(model_a, X_train, y_train, list(X_train.columns))       
        return self.trained_model

# Trained model
class TrainedModelA:                                               
    def __init__(self, model_a, X_train, y_train, feature_names):  
        self.model = model_a                                       
        self.X_train = X_train                                     
        self.y_train = y_train                                     
        self.feature_names = feature_names                         
        self.perturbator = PerturbatorA(self)                       
        print("TrainedModelA instantiated.")

    def predict(self, X):                                          
        return self.model.predict(X)                               

# Perturbator
class PerturbatorA:
    def __init__(self, trained_model_a):
        self.trained_model = trained_model_a
        self.explainer = LimeExplainerA(self)
        print("PerturbatorA created and connected to TrainedModelA.")

    def perturb(self, instance, num_samples=50):
        import numpy as np
        import pandas as pd

        print("Perturbing...")
        perturbed_data = []
        for _ in range(num_samples):
            noise = np.random.normal(0, 0.1, size=instance.shape)
            perturbed_instance = instance + noise
            perturbed_data.append(perturbed_instance)
        
        df_perturbed = pd.DataFrame(perturbed_data, columns=self.trained_model.feature_names)
        print(f"{num_samples} perturbed instances.")
        return df_perturbed

# Lime Explainer
class LimeExplainerA:
    def __init__(self, perturbator_a):
        self.perturbator = perturbator_a
        print("LimeExplainerA instantiated and connected to PerturbatorA.")

    def explain_instance(self, instance):
        from lime.lime_tabular import LimeTabularExplainer
        import numpy as np

        print("Executing LIME explication.")

        model_a = self.perturbator.trained_model.model
        X_train = self.perturbator.trained_model.X_train
        feature_names = self.perturbator.trained_model.feature_names

        explainer_a = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=feature_names,
            class_names=['malignant', 'benign'],
            mode="classification"
        )

        instance_array = np.array(instance).reshape(1, -1)
        explanation_a = explainer_a.explain_instance(
            data_row=instance_array[0],
            predict_fn=model_a.predict_proba
        )

        explanation_a.show_in_notebook(show_table=True)
        return explanation_a


# *** Model B - RF ***

# Untrained model
class ModelB_RF:
    def __init__(self, database: ClinicalDatabase):
        self.model_type = "RandomForest"
        self.database = database
        self.trained_model = None

    def build_model(self):
        X = self.database.get_features(self.database.feature_names)
        y = self.database.get_labels()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model_b = RandomForestClassifier(n_estimators=100, random_state=42)
        model_b.fit(X_train, y_train)
        print("ModelB_RF successfully trained.")
        self.trained_model = TrainedModelB(model_b, X_train, y_train, list(X_train.columns))
        return self.trained_model

# Trained model
class TrainedModelB:
    def __init__(self, model_b, X_train, y_train, feature_names):
        self.model = model_b
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.perturbator = PerturbatorB(self)
        print("TrainedModelB instantiated.")

    def predict(self, X):
        return self.model.predict(X)  

# Perturbator
class PerturbatorB:
    def __init__(self, trained_model_b):
        self.trained_model = trained_model_b
        self.explainer = LimeExplainerB(self)
        print("PerturbatorB created and connected to TrainedModelB.")  

    def perturb(self, instance, num_samples=50):
        import numpy as np
        import pandas as pd

        print("Perturbing...")
        perturbed_data = []
        for _ in range(num_samples):
            noise = np.random.normal(0, 0.1, size=instance.shape)
            perturbed_instance = instance + noise
            perturbed_data.append(perturbed_instance)

        df_perturbed = pd.DataFrame(perturbed_data, columns=self.trained_model.feature_names)
        print(f"{num_samples} perturbed instances.")
        return df_perturbed

# Lime Explainer
class LimeExplainerB:
    def __init__(self, perturbator_b):
        self.perturbator = perturbator_b
        print("LimeExplainerB instantiated and connected to PerturbatorB.")

    def explain_instance(self, instance):
        from lime.lime_tabular import LimeTabularExplainer
        import numpy as np

        print("Executing LIME explication.")

        model_b = self.perturbator.trained_model.model
        X_train = self.perturbator.trained_model.X_train
        feature_names = self.perturbator.trained_model.feature_names

        explainer_b = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=feature_names,
            class_names=['malignant', 'benign'],
            mode="classification"
        )

        instance_array = np.array(instance).reshape(1, -1)
        explanation_b = explainer_b.explain_instance(
            data_row=instance_array[0],
            predict_fn=model_b.predict_proba
        )

        explanation_b.show_in_notebook(show_table=True)
        return explanation_b


# *** Model C - LogReg ***

# Untrained model
class ModelC_LogReg:
    def __init__(self, database: ClinicalDatabase):
        self.model_type = "LogisticRegression"
        self.database = database
        self.trained_model = None

    def build_model(self):
        X = self.database.get_features(self.database.feature_names)
        y = self.database.get_labels()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model_c = LogisticRegression(max_iter=5000, random_state=42)
        model_c.fit(X_train, y_train)
        print("ModelC_LogReg successfully trained.")
        self.trained_model = TrainedModelC(model_c, X_train, y_train, list(X_train.columns))
        return self.trained_model

# Trained model
class TrainedModelC:
    def __init__(self, model_c, X_train, y_train, feature_names):
        self.model = model_c
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.perturbator = PerturbatorC(self)
        print("TrainedModelC instantiated.")

    def predict(self, X):
        return self.model.predict(X)

# Perturbator
class PerturbatorC:
    def __init__(self, trained_model_c):
        self.trained_model = trained_model_c
        self.explainer = LimeExplainerC(self)
        print("PerturbatorC created and connected to TrainedModelC.")

    def perturb(self, instance, num_samples=50):
        import numpy as np
        import pandas as pd

        print("Perturbing...")
        perturbed_data = []
        for _ in range(num_samples):
            noise = np.random.normal(0, 0.1, size=instance.shape)
            perturbed_instance = instance + noise
            perturbed_data.append(perturbed_instance)

        df_perturbed = pd.DataFrame(perturbed_data, columns=self.trained_model.feature_names)
        print(f"{num_samples} perturbed instances.")
        return df_perturbed

# Lime Explainer
class LimeExplainerC:
    def __init__(self, perturbator_c):
        self.perturbator = perturbator_c
        print("LimeExplainerC instantiated and connected to PerturbatorC.")

    def explain_instance(self, instance):
        from lime.lime_tabular import LimeTabularExplainer
        import numpy as np

        print("Executing LIME explication.")

        model_c = self.perturbator.trained_model.model
        X_train = self.perturbator.trained_model.X_train
        feature_names = self.perturbator.trained_model.feature_names

        explainer_c = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=feature_names,
            class_names=['malignant', 'benign'],
            mode="classification"
        )

        instance_array = np.array(instance).reshape(1, -1)
        explanation_c = explainer_c.explain_instance(
            data_row=instance_array[0],
            predict_fn=model_c.predict_proba
        )

        explanation_c.show_in_notebook(show_table=True)
        return explanation_c


# *** Main ***

if __name__ == "__main__":
    db = ClinicalDatabase()

    # ModelA Explication
    model_a = ModelA_MLP(db)
    trained_model_a = model_a.build_model()
    sample_instance_a = trained_model_a.X_train.iloc[0]
    print("Selected instance:")
    print(sample_instance_a)

    print("Instance real target:")
    print(trained_model_a.y_train.iloc[0])

    print("Model prediction for the instance:")
    print(trained_model_a.predict(sample_instance_a.to_frame().T))

    print("Model Accuracy:")
    print(trained_model_a.model.score(trained_model_a.X_train, trained_model_a.y_train))

    print("MLP architecture:")
    print(trained_model_a.model.hidden_layer_sizes)

    trained_model_a.perturbator.explainer.explain_instance(sample_instance_a)

    
    # ModelB Explication
    model_b = ModelB_RF(db)
    trained_model_b = model_b.build_model()
    sample_instance_b = trained_model_b.X_train.iloc[0]

    print("Selected instance:")
    print(sample_instance_b)
    
    print("Instance real target:")
    print(trained_model_b.y_train.iloc[0])

    print("Model prediction for the instance:")
    print(trained_model_b.predict(sample_instance_b.to_frame().T))

    print("Model Accuracy:")
    print(trained_model_b.model.score(trained_model_b.X_train, trained_model_b.y_train))

    trained_model_b.perturbator.explainer.explain_instance(sample_instance_b)

    
    # ModelC Explication
    model_c = ModelC_LogReg(db)
    trained_model_c = model_c.build_model()
    sample_instance_c = trained_model_c.X_train.iloc[0]

    print("Selected instance:")
    print(sample_instance_c)

    print("Instance real target:")
    print(trained_model_c.y_train.iloc[0])

    print("Model prediction for the instance:")
    print(trained_model_c.predict(sample_instance_c.to_frame().T))

    print("Model Accuracy:")
    print(trained_model_c.model.score(trained_model_c.X_train, trained_model_c.y_train))

    trained_model_c.perturbator.explainer.explain_instance(sample_instance_c)
