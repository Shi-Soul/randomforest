import os
import time
import logging
import pickle
import matplotlib.pyplot as plt
import argparse
import numpy as np

def parse(arg):
    parser = argparse.ArgumentParser()

        
    parser.add_argument("-V", "--verbose", help="increase output verbosity",
                action="store_true")
    parser.add_argument("-D", "--debug", help="print debug info",action="store_true")
    parser.add_argument("-S","--sklearn", help="Use sklearn's RandomForest",action="store_true")
    parser.add_argument("-p","--parallel", help="Disable Parallel to speed up",default=True,action="store_false")
    # parser.add_argument("-p","--parallel", help="Disable Parallel to speed up",default=False,action="store_false")
    parser.add_argument("--save", help="save model path",type=str, default="./save")


    parser.add_argument("-d","--dataset", help="dataset name", type=str, default="mnist",
                choices=["mnist","cifar10","test"])
    parser.add_argument("--val_ratio", help="validation ratio", type=float, default=0.1)
    # parser.add_argument("--rf_type", help="validation ratio", type=str, default="rf",\
                        # choices=["rf","ccf"])
    parser.add_argument("--rf_type", help="validation ratio", type=str, default="rf",\
                        choices=["rf","ccf"])


        # RandomForestClassifier parameters
    # parser.add_argument("--random_state", help="random state",type=int, default=42)
    parser.add_argument("--n_jobs", help="number of jobs",type=int, default=4)
    parser.add_argument("--soft_pred", help="When predicting in a leaf, return hard or soft prediction.", \
                        action="store_true",default=False)
    parser.add_argument("--criterion", help="criterion",type=str, \
                        default="gini",choices=["gini","entropy","log_loss"])
    parser.add_argument("--n_estimators", help="number of estimators",type=int, default=10)
    parser.add_argument("--max_depth", help="max depth",type=int, default=None)
    parser.add_argument("--min_samples_split", help="min samples split",type=int, default=2)
    parser.add_argument("--min_samples_leaf", help="min samples leaf",type=int, default=1)
    parser.add_argument("--max_features", help="max features",type=float, default=None)
    parser.add_argument("--max_leaf_nodes", help="max leaf nodes",type=int, default=None)
    parser.add_argument("--min_impurity_decrease", help="min impurity decrease",type=float, default=0.0)
    parser.add_argument("--bootstrap", help="bootstrap",type=bool, default=True)
    parser.add_argument("--projection_bootstrap", help="projection_bootstrap",\
                        type=bool, default=True)

    parser.add_argument("--oob_score", help="oob score",type=bool, default=False)
    # parser.add_argument("--warm_start", help="warm start",type=bool, default=False)
    # parser.add_argument("--class_weight", help="class weight",type=str, default=None)
    parser.add_argument("--ccp_alpha", help="ccp alpha",type=float, default=0.0)
    parser.add_argument("--max_samples", help="max samples",type=float, default=None)


    config = parser.parse_args(args=arg.split())
    if config.debug:
        config.verbose = True
    return config



def get_model_config(config):
    model_config = {
        # "random_state":config.random_state,
        "verbose":int(config.verbose),
        "criterion":config.criterion,
        "n_estimators":config.n_estimators,
        "max_depth":config.max_depth,
        "n_jobs":config.n_jobs,
        "min_samples_split":config.min_samples_split,
        "min_samples_leaf":config.min_samples_leaf,
        "max_features":config.max_features,
        "max_leaf_nodes":config.max_leaf_nodes,
        "min_impurity_decrease":config.min_impurity_decrease,
        "bootstrap":config.bootstrap,
        "oob_score":config.oob_score,
        # "warm_start":config.warm_start,
        # "class_weight":config.class_weight,
        "ccp_alpha":config.ccp_alpha,
        "max_samples":config.max_samples,
    }
    return model_config

def plot_test(data,label,pred=None,clf=None):
    if clf is not None:
        plt.figure(figsize=(5, 5))
        plt.subplot(1,1,1)
        plt.scatter(data[:,0],data[:,1],c=label)
        ## plot decision boundary
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4)
    elif pred is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
        plt.scatter(data[:,0],data[:,1],c=label)
        plt.subplot(1,2,2)
        plt.scatter(data[:,0],data[:,1],c=pred)
    plt.show()

def get_test():
    ## Get Test Dataset
    ## 5 classes, 2 features, 100 samples; Gaussian

    np.random.seed(52)
    num_class = 6
    num_feature = 2
    num_sample = 500
    mean = np.array([ [0.4,1],[-7,2],[3,-2],[5,8],[-3,-4],[-2,6] ])
    # mean = np.random.randn(num_class,num_feature)*10
    cov = np.random.randn(num_class,num_feature,num_feature)
    cov = np.matmul(cov.transpose(0,2,1),cov)*0.5+ np.eye(num_feature)*1e-3
    dataset = []
    for i in range(num_class):
        dataset.append(np.random.multivariate_normal(mean[i],cov[i],num_sample))
    dataset = np.concatenate(dataset,axis=0)
    labels = np.concatenate([np.ones(num_sample)*i for i in range(num_class)],axis=0)
    idx = np.arange(dataset.shape[0])
    np.random.shuffle(idx)
    dataset = dataset[idx]
    labels = labels[idx]


    val_ratio = 0.2
    num = dataset.shape[0]
    val_num = int(num*val_ratio)
    train_num = num - val_num
    train_dataset = []
    val_dataset = []
    for term in [dataset,labels]:
        train_dataset.append(term[:train_num])
        val_dataset.append(term[train_num:])
    return train_dataset[0],train_dataset[1],val_dataset[0],val_dataset[1]
    pass

def get_dataset(config):
    if config.dataset =="mnist":
        from mnist import load_mnist
        dataset = load_mnist()
    elif config.dataset =="cifar10":
        from cifar10 import get_small,read_small
        # dataset = get_small()
        dataset = read_small()
    elif config.dataset =="test":
        dataset = get_test()

    dataset_shape = dataset[0].shape[1:]
    _dataset = []
    for i,(term) in enumerate(dataset):
        term = term.reshape(term.shape[0],-1)
        _dataset.append(term) 
        if config.debug:
            ## Get 500 samples for debug
            _dataset[i] = term[:1000]
    return _dataset,dataset_shape
    # print(term[0].shape,term[1].shape)

def split_dataset(dataset,val_ratio):
    num = dataset[0].shape[0]
    val_num = int(num*val_ratio)
    train_num = num - val_num
    train_dataset = []
    val_dataset = []
    train_images,train_labels,test_images,test_labels = dataset
    for term in [train_images,train_labels]:
        train_dataset.append(term[:train_num])
        val_dataset.append(term[train_num:])
    return train_dataset,val_dataset,[test_images,test_labels]

def show_images(images,labels,dataset_shape):
    row_col_show = (2,5)
    num_show = row_col_show[0]*row_col_show[1]

    images = images.reshape(-1,*dataset_shape)
    print(labels[:num_show],labels[:num_show].shape,labels[:num_show].dtype)
    images = images.transpose(0,2,3,1)
    plt.figure(figsize=(10, 5))
    for i in range(num_show):
        plt.subplot(row_col_show[0],row_col_show[1],i+1)
        plt.imshow(images[i])
    plt.show()



def sklearn_rf(model_config):
    import sklearn
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(**model_config)
    return model
# def lightgbm_rf(model_config):
#     import lightgbm as lgb
#     model = lgb.LGBMClassifier(**model_config)
#     return model

def train_model(config,model,train_dataset,val_dataset):

    # Train Model
    start_time = time.time()
    train_images,train_labels = train_dataset
    val_images,val_labels = val_dataset

    model.fit(train_images, train_labels)

    score = model.score(val_images, val_labels)
    end_time = time.time()

    # Save Model
    save_path = os.path.join(config.save,\
                             f"{'my' if not config.sklearn else 'sk' }_{config.dataset}_{ int(time.time())%(int(1e7)) }.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)

    # Print Info
    info =  f"\nFinishing Training RandomForest\n"+\
            f"Model Saved to {save_path}\n"+\
            f"Validation Accuracy: {score:.4f}, Validation Samples Num: {val_labels.shape[0]}\n"
    info += f"Time Cost: {end_time-start_time:.2f}s"
    print(info)
    logging.info(info)
    # return model



def get_model(config,model_config):
    if config.sklearn:
        model = sklearn_rf(model_config)
        # model = lightgbm_rf(model_config)
    else:
        if config.rf_type=="rf":
            from model import RandomForest
            model = RandomForest(config,model_config)
        elif config.rf_type=="ccf":
            from model_ccf import CCF
            model = CCF(config,model_config)
    return model