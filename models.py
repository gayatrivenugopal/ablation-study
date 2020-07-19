from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier
import xgboost as xgb

class Models:
    def get_model(model_num):
        """
        Return the model based on the model number.
        Argument:
        model_num (int)
        """
        if model_num == 1:
            return RandomForestClassifier(n_estimators=100, max_features="auto",random_state=0)
        elif model_num == 2:
            return ExtraTreesClassifier(n_estimators=100, max_features="auto",random_state=0)
        elif model_num == 3:
            return AdaBoostClassifier(n_estimators=100)
        elif model_num == 4:
            return GradientBoostingClassifier(n_estimators=100)
        elif model_num == 5:
            return xgb.XGBClassifier(random_state=1,learning_rate=0.01)
    
    def get_model_name(model_num):
        """
        Return the name of the model based on the model number.
        Argument:
        model_num (int)
        """
        if model_num == 1:
            return 'RandomForestClassifier'
        elif model_num == 2:
            return 'ExtraTreesClassifier'
        elif model_num == 3:
            return 'AdaBoostClassifier'
        elif model_num == 4:
            return 'GradientBoostingClassifier'
        elif model_num == 5:
            return 'XGBClassifier'
