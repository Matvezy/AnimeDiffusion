import os
from time import gmtime, strftime
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
import yaml

sagemaker_role = "*MyRole*"

# Training Data Upload to S3
"""
def upload_training_data(session,base_job_name):
    traindata_s3_prefix = '{}/data/train'.format(base_job_name)
    train_s3 = session.upload_data(path='train', key_prefix=traindata_s3_prefix)
    return train_s3
"""

def sagemaker_estimator(sagemaker_role, code_entry, code_dir, instance_type, instance_count, hyperparameters, metric_definitions):
    estimator = PyTorch(entry_point=code_entry,
                    source_dir=code_dir,
                    role=sagemaker_role,
                    py_version='py38',
                    instance_type=instance_type,
                    instance_count=instance_count,
                    metric_definitions=metric_definitions,
                    framework_version='1.10.2',
                    hyperparameters=hyperparameters,
                    script_mode=True)
    return estimator


def sagemaker_training(sm_estimator,train_s3,training_job_name):
    sm_estimator.fit(train_s3, job_name=training_job_name, wait=False)


if __name__ == '__main__':
    train_s3 = "*location*"
    code_entry = 'train.py'
    code_dir = os.getcwd()
    instance_type = 'ml.g5.xlarge'
    instance_count = 1
    hyperparameters = {'epochs': 200, 'batch_size': 16, 'batch_size_val': 6, 'learning_rate': 3e-4, 'dropout': 0.2}
    metric_definitions = [
        {'Name': 'epoch', 'Regex': 'Epoch: (.*?);'},
        {'Name': 'train:error', 'Regex': 'Train Loss: (.*?);'},
        {'Name': 'validation:error', 'Regex': 'Valid Loss: (.*?);'}   
    ]
    sm_estimator = sagemaker_estimator(sagemaker_role, code_entry, code_dir, instance_type, instance_count, hyperparameters, metric_definitions)
    print(sm_estimator)
        
    training_job_name = "test-training-{}".format(strftime("%d-%H-%M-%S", gmtime()))
    print(training_job_name)
    model_dir = "s3://sagemaker-us-east-1-104995299079/" + training_job_name + "/output/model.tar.gz"
    
    with open("config.yaml") as f:
        list_doc = yaml.safe_load(f)

    list_doc["DEPLOY"]["S3_PATH"] = model_dir

    with open("config.yaml", "w") as f:
        yaml.dump(list_doc, f)
    sagemaker_training(sm_estimator, train_s3, training_job_name)