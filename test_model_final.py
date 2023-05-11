
import json
import sys

import joblib
import pandas as pd
from fairlearn.metrics import MetricFrame, false_negative_rate
from sklearn.metrics import balanced_accuracy_score

USAGE = """%s [model_file] [test_file] [config_file] [output_file]
""" % sys.argv[0]

def test_model_fairness(model_file, test_file, config_file, metrics_file):

    # Loading configuration
    configFile = json.loads(open(config_file).read())
    sensitiveFeatureNames = configFile['sensitiveFeatures']
    # e.g. {"sensitiveFeatures" : "race","threshold":0.1}
    
    # Loading test data
    testData = pd.read_csv(test_file, index_col=False)

    Y_test = testData.iloc[:,-2]
    X_test = testData.iloc[:,0:-2]
    sensitiveFeatures = testData[sensitiveFeatureNames]

    # Loading ML model 
    model=joblib.load(model_file)
    Y_pred = model.predict(X_test)
   
    # Fairness assessment    
    balanced_accuracy_score(Y_test, Y_pred)
    mf = MetricFrame(metrics=false_negative_rate,
                      y_true=Y_test,
                      y_pred=Y_pred,
                      sensitive_features=sensitiveFeatures)

    dictionary = {'difference': str(mf.difference()),
                  'ratio': str(mf.ratio()),
                  'group_min': str(mf.group_min()),
                  'group_max': str(mf.group_max())}

    model_metrics = json.dumps(dictionary) 
    print(model_metrics)

    f = open(metrics_file, 'wt')
    f.write(model_metrics)
    f.close()

    # Check threshold to proceed to the next step
    if float(dictionary['difference'])> configFile["threshold"]:
        print('Model is not fair')
        sys.exit(1)
    else:
        print('Model is fair')

if __name__ == '__main__':
    if len(sys.argv) == 5:
        model_file = sys.argv[1]
        test_file = sys.argv[2]
        config_file = sys.argv[3]
        metrics_file = sys.argv[4]
        test_model_fairness(model_file, test_file, config_file, metrics_file)
    else:
        print(USAGE)
