# Random Forest Model
# Hard Predict, No Soft Predict
import numpy as np
import math
import time
import logging
from joblib import Parallel, delayed
# from utils import *


def resample(images,labels,N_samples,rdm=None):
    ## N= labels.shape[0], N_samples : resample number
    if rdm:
        # np.random.seed(rdm)
        # new_index = rdm.choice(np.arange(len(labels)),N_samples)
        new_index = np.random.choice(np.arange(len(labels)),N_samples)
    else:
        new_index = np.random.choice(np.arange(len(labels)),N_samples)
    new_images = images[new_index]
    new_labels = labels[new_index]
    return new_images,new_labels

def gini(labels):
    # return : gini
    I,C = np.unique(labels,return_counts=True, axis=0)
    gini = 1 - np.sum((C/C.sum())**2)
    return gini
    pass

def cond_entropy(data,labels,feature):
    # return : conditonal_entropy
    N = data.shape[0]
    X = data[:,feature]
    X_Y = np.concatenate((X.reshape(-1,1),labels.reshape(-1,1)),axis=1)

    H_X_Y = entropy(X_Y)
    H_X = entropy(X)

    cond_entropy = H_X_Y - H_X
    return cond_entropy

def entropy(labels):
    # return : entropy
    I,C = np.unique(labels,return_counts=True, axis=0)
    # C = np.bincount(labels)
    p = C/C.sum()
    entropy = -np.sum(p*np.log2(p))
    return entropy
    pass


class TreeNode(): 
    def __init__(self,config,model_config,tree_config,depth=0,features_set=None):
        self.config = config
        self.model_config = model_config
        self.tree_config = tree_config
        self.rdm = self.tree_config["rdm"]
        
        # self.n_tree_samples = None
        # self.n_node_samples = None
        self.depth = depth
        self.child = []
        self.is_leaf = False
        if features_set is None:
            self.features_set = list(range(self.tree_config["n_features"]))
        else:
            self.features_set = features_set
        self.split_feature = None
        self.split_value = None
        self.split_gain = None
        self.split_feature_value_gain= None
        self.predict_value = None

    def fit(self,train_images,train_labels):
        self.n_node_samples = train_images.shape[0]
        ## Sort Data

        # Exclude Edge Cases
        if (self.model_config["max_depth"] is not None and 
            self.depth >= self.model_config["max_depth"]) or \
            self.n_node_samples < self.model_config["min_samples_split"] or \
            self.features_set == None or \
            np.unique(train_images[:,self.features_set],axis=0).shape[0] == 1:
            ## Too deep or too few samples or no features or no split

            # self.is_leaf = True
            # I,C = np.unique(train_labels,return_counts=True, axis=0)
            # self.predict_value = I[np.argmax(C)]
            self.be_leaf(train_labels)
            return self
        if len(np.unique(train_labels)) == 1:
            # self.is_leaf = True
            # self.predict_value = train_labels[0]
            self.be_leaf(train_labels)
            return self

        self.gain_base = self._criterion(train_labels)
        # Choose split feature and split value
        if self.model_config["max_features"]==None:
            self.m_features = int(np.sqrt(self.tree_config["n_features"]))
        else:
            self.m_features = int(self.model_config["max_features"]*self.tree_config["n_features"])
        
        self.split_feature = self.choose_split_feature(train_images,train_labels,self.m_features)
        
        sort_ind = np.argsort(train_images[:,self.split_feature])
        train_images = train_images[sort_ind]
        train_labels = train_labels[sort_ind]
        self.split_value,self.split_gain = \
            self.choose_split_value(train_images,train_labels,self.split_feature)
        self.real_gain = (self.split_gain)\
                            *self.n_node_samples/self.tree_config["n_tree_samples"] 

        # Split
        left_images,left_labels,right_images,right_labels,left_index,right_index = \
            self.split(feature=self.split_feature,split_value=self.split_value,\
                        train_images=train_images,train_labels=train_labels)
        # Exclude Edge Cases
        if left_images.shape[0] <= self.tree_config["min_samples_leaf"] or \
            right_images.shape[0] <= self.tree_config["min_samples_leaf"] or \
            self.real_gain  < self.model_config["min_impurity_decrease"]:

            # self.is_leaf = True
            # I,C = np.unique(train_labels,return_counts=True, axis=0)
            # self.predict_value = I[np.argmax(C)]
            self.be_leaf(train_labels)
            return self
        
        # Build Child
        features_set = self.features_set.copy().remove(self.split_feature)
        self.child.append(TreeNode(self.config,self.model_config,\
                                   self.tree_config,self.depth+1,features_set))
        self.child.append(TreeNode(self.config,self.model_config,\
                                   self.tree_config,self.depth+1,features_set))
        self.child[0].fit(left_images,left_labels)
        self.child[1].fit(right_images,right_labels)

        # Check node attr
        if self.is_leaf:
            assert self.is_leaf == True and len(self.child) == 0 and self.predict_value is not None, \
            f"A leaf node should have no child and a predict value, {self.is_leaf, len(self.child), self.predict_value}"
        else:
            assert self.is_leaf == False and len(self.child) == 2 and \
            self.split_feature is not None and self.split_value is not None, \
            f"A non-leaf node should have two child and a split feature and a split value,"+\
            f" {self.is_leaf, len(self.child), self.split_feature, self.split_value}"
        
        return self

    def predict(self,test_images):
        if self.is_leaf:
            return self.predict_value
        else:
            left_images,left_labels,right_images,right_labels,left_index,right_index = \
                self.split(self.split_feature,self.split_value,test_images,None)
            left_pred = self.child[0].predict(left_images)
            right_pred = self.child[1].predict(right_images)
            if not self.config.soft_pred:
                pred = np.zeros(test_images.shape[0])
                pred[left_index] = left_pred
                pred[right_index] = right_pred
            else:
                pred = np.zeros((test_images.shape[0],self.tree_config["n_classes"]))
                pred[left_index] = left_pred
                pred[right_index] = right_pred
            return pred

    def be_leaf(self,train_labels):
        self.is_leaf = True
        I,C = np.unique(train_labels,return_counts=True, axis=0)
        if not self.config.soft_pred:
            # predict_value: (1), class
            self.predict_value = I[np.argmax(C)]
        else:
            # predict_value: (n_classes), probability
            predict_bin = C/C.sum()
            predict_bin = predict_bin.reshape(-1)
            predict_value = np.zeros(self.tree_config["n_classes"])
            predict_value[I.astype(np.int32).reshape(-1)] = predict_bin
            self.predict_value = predict_value
        return self

    def choose_split_feature(self,train_images,train_labels,m_features):
        choices = self.rdm.choice(self.features_set,m_features,replace=False)
        choices = np.random.choice(self.features_set,m_features,replace=False)
        assert len(choices)==m_features,f"choices: {choices}, len_choices-m: {len(choices)-m_features}"
        best_feature = None
        best_gain = -np.inf
        criterion = self.model_config["criterion"]
        for feature in choices:
            gain = self.get_split_feature_gain(train_images,train_labels,feature,criterion)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        return best_feature
    
    def get_split_feature_gain(self,train_images,train_labels,feature,criterion):
        # return : gain
        if criterion=="gini":
            if self.split_feature_value_gain is None:
                self.split_feature_value_gain = {}
            # 计算这个feature对应的所有组合中gain最大的, 保留下来, 后续可以直接调用
            sort_ind = np.argsort(train_images[:,feature])
            train_images = train_images[sort_ind]
            train_labels = train_labels[sort_ind]
            self.split_feature_value_gain[feature] = self.choose_split_value(train_images,train_labels,feature)
            return self.split_feature_value_gain[feature][1]
            # return self.get_gini_gain(train_images,train_labels,f
        elif criterion=="entropy" or criterion=="log_loss":
            return entropy(train_labels)- cond_entropy(train_images,train_labels,feature)

        
    def choose_split_value(self,train_images,train_labels,feature): 
        # return : split_value
        criterion = self.model_config["criterion"]
        I,C = np.unique(train_images[:,feature],return_counts=True, axis=0) # It's a sorted array, Y is also sorted

        # if len(I) == 1:return I[0],self.get_split_gain(I,C,train_labels,I[0],criterion)
        if len(I) == 1:return I[0],0
        
        split_value = (I[1:]+I[:-1])/2
        split_gains = np.zeros(len(split_value))

    # Linear Search
        # for i in range(len(split_value)):
        #     split_gains[i] = self.get_split_gain(I,C,train_labels,split_value[i],criterion)
        # __best_index = np.argmax(split_gains)
        # best_index = np.argmax(split_gains)
        # return split_value[best_index],split_gains[best_index]

    # Binary Search 
        left_index = 0
        right_index = len(split_value)-1
        split_gains[left_index] = self.get_split_gain(I,C,train_labels,split_value[left_index],criterion)
        split_gains[right_index] = self.get_split_gain(I,C,train_labels,split_value[right_index],criterion)
        best_index = 0
        best_gain = split_gains[0]
        while left_index < right_index:
            mid_index = (left_index+right_index)//2
            if split_gains[mid_index] ==0 :
                split_gains[mid_index] = self.get_split_gain(I,C,train_labels,split_value[mid_index],criterion)
            if split_gains[mid_index+1] ==0 :
                split_gains[mid_index+1] = self.get_split_gain(I,C,train_labels,split_value[mid_index+1],criterion)
            if split_gains[mid_index-1] ==0 :
                split_gains[mid_index-1] = self.get_split_gain(I,C,train_labels,split_value[mid_index-1],criterion)
            
            if split_gains[mid_index] > split_gains[mid_index+1] and split_gains[mid_index] > split_gains[mid_index-1]:
                right_index = mid_index
                left_index = mid_index
            elif split_gains[mid_index] > split_gains[mid_index+1] and split_gains[mid_index] < split_gains[mid_index-1]:
                right_index = mid_index-1
            elif split_gains[mid_index] < split_gains[mid_index+1] and split_gains[mid_index] > split_gains[mid_index-1]:
                left_index = mid_index+1
            else:
                if split_gains[mid_index+1] > split_gains[mid_index-1]:
                    left_index = mid_index+1
                else:
                    right_index = mid_index-1
            if split_gains[mid_index] > best_gain:
                best_index = mid_index
                best_gain = split_gains[mid_index]
            if split_gains[mid_index+1] > best_gain:
                best_index = mid_index+1
                best_gain = split_gains[mid_index+1]
            if split_gains[mid_index-1] > best_gain:
                best_index = mid_index-1
                best_gain = split_gains[mid_index-1]
        # assert best_index == __best_index, \
        #         f"best_index: {best_index}, __best_index: {__best_index}, split_gains: {list(enumerate(split_gains))}"
        return split_value[best_index],best_gain

    def get_split_gain(self,I,C,Y,split_value,criterion): 
                # return : gain
        left_index = C[I<=split_value].sum()
        left_Y = Y[:left_index]
        right_Y = Y[left_index:]
        return self.gain_base - (left_Y.shape[0]/Y.shape[0])*self._criterion(left_Y) - (right_Y.shape[0]/Y.shape[0])*self._criterion(right_Y)
    
    def split(self,feature,split_value,train_images,train_labels=None):
        # return : left_images,left_labels,right_images,right_labels
        left_index = train_images[:,feature] <= split_value
        right_index = train_images[:,feature] > split_value
        
        left_images = train_images[left_index]
        right_images = train_images[right_index]

        if train_labels is not None:
            left_labels = train_labels[left_index]  
            right_labels = train_labels[right_index]
        else:
            left_labels = None
            right_labels = None

        return left_images,left_labels,right_images,right_labels,left_index,right_index
        pass

    def _get_tree_depth(self):
        if self.is_leaf:
            return 0
        else:
            return 1+max(self.child[0]._get_tree_depth(),self.child[1]._get_tree_depth())
        
    def _get_tree_nodes(self):
        if self.is_leaf:
            return 1
        else:
            return 1+self.child[0]._get_tree_nodes()+self.child[1]._get_tree_nodes()

    def _criterion(self,Y):
        if self.model_config["criterion"]=="gini": 
            return gini(Y)
        elif self.model_config["criterion"]=="entropy" or self.model_config["criterion"]=="log_loss":
            return entropy(Y)

class BaseDecisionTree(): 
    def __init__(self,config,model_config,tree_config):
        self.config = config
        self.model_config = model_config
        self.tree_config = tree_config

        self.root = None

    def fit(self,train_images,train_labels):
        self.tree_config["n_tree_samples"] = train_images.shape[0]
        self.root = TreeNode(self.config,self.model_config,self.tree_config)
        self.root.fit(train_images,train_labels)
        return self
        pass

    def predict(self,test_images):
        return self.root.predict(test_images)

    def _get_tree_depth(self):
        return self.root._get_tree_depth()
    
    def _get_tree_nodes(self):
        return self.root._get_tree_nodes()

class RandomForest():
    def __init__(self,config,model_config):
        self.config = config
        self.model_config = model_config

        # self.rdm = np.random.RandomState(model_config["random_state"]) 
        self.rdm = np.random.RandomState(1) 
        self.n_jobs = model_config["n_jobs"]
        self.parallel = self.config.parallel
        self.verbose = model_config["verbose"]

        self.bootstrap = model_config["bootstrap"]
        self.criterion = model_config["criterion"]
        self.n_estimators = model_config["n_estimators"] # int
        self.max_samples = model_config["max_samples"] # float
        

        self.trees = []
        self.n_train_samples = None
        self.n_features = None
        self.n_classes = None
        
        pass

    def fit(self,train_images,train_labels):
        start_time = time.time()

        __train_images = train_images.reshape(train_images.shape[0],-1)
        __train_labels = train_labels.reshape(train_labels.shape[0],-1)
        self.n_train_samples, self.n_features = __train_images.shape
        self.n_classes = len(np.unique(train_labels))

        if self.max_samples is not None :
            n_tree_samples = int(self.n_train_samples*self.max_samples)
        else:
            n_tree_samples = self.n_train_samples

        self.tree_config = {
            "rdm":self.rdm,
            "n_tree_samples":n_tree_samples,
            "n_features":self.n_features,
            "n_classes":self.n_classes,

            "criterion":self.model_config["criterion"],
            "max_depth":self.model_config["max_depth"],
            "min_samples_split":self.model_config["min_samples_split"],
            "min_samples_leaf":self.model_config["min_samples_leaf"],
            "max_features":self.model_config["max_features"],
            "max_leaf_nodes":self.model_config["max_leaf_nodes"],
            "min_impurity_decrease":self.model_config["min_impurity_decrease"],
            "ccp_alpha":self.model_config["ccp_alpha"],

        }

        if self.verbose:
            info = "Creating the Random Forest\n"
            info += f"train_images: {train_images.shape }, train_labels: {train_labels.shape}"
            print(info)
            logging.info(info)
        
        if self.parallel:
            self.trees = Parallel(n_jobs=self.n_jobs,verbose=self.verbose)(delayed(self.make_bdt)\
                (self.config,self.model_config,self.tree_config,i,__train_images,__train_labels,n_tree_samples)\
                for i in range(self.n_estimators))
        else:
            
            self.trees = [self.make_bdt(self.config,self.model_config,self.tree_config,\
                                        i,__train_images,__train_labels,n_tree_samples)\
                        for i in range(self.n_estimators)]

            # for i in range(self.n_estimators):
            #     if self.verbose and i%math.ceil(self.n_estimators/5)==0:
            #         info = f"Creating the {i+1}/{self.n_estimators} Base Decision Tree"
            #         print(info)
            #         logging.info(info)
            #     if self.bootstrap:
            #         train_images,train_labels = resample(__train_images ,__train_labels,n_tree_samples,self.rdm)
            #     else:
            #         train_images,train_labels = __train_images ,__train_labels
            #     tree = BaseDecisionTree(self.config,self.model_config,self.tree_config)
            #     tree.fit(train_images,train_labels)
            #     self.trees.append(tree)
            #     if self.verbose and i%math.ceil(self.n_estimators/5)==0:
            #         info = f"Finish Creating the {i+1}/{self.n_estimators} Base Decision Tree"+ \
            #             f" with depth {tree._get_tree_depth()}, n_nodes {tree._get_tree_nodes()}\n"
            #         info += f" time per tree: {round((time.time()-start_time)/(i+1),3)} s"+\
            #             f" total time: {round(time.time()-start_time,3)} s,"+\
            #             f" time left: {round((time.time()-start_time)*(self.n_estimators-i-1)/(i+1),3)} s"
            #         print(info)
            #         logging.info(info)

        if self.verbose:
            info = f"Finish Creating the Random Forest with {self.n_estimators} Base Decision Trees"+\
                f" with depth:\n\t {[tree._get_tree_depth() for tree in self.trees]}\n\t"+\
                f" with num of nodes: {[tree._get_tree_nodes() for tree in self.trees]}\n"
            info += f" total time: {round(time.time()-start_time,3)} s"
            print(info)
            logging.info(info)

        # TODO: pruning trees
        return self

    def predict_proba(self,test_images):
        n_test_samples = test_images.shape[0]
        test_images = test_images.reshape(n_test_samples,-1)
        if not self.config.soft_pred:
            predictions = np.zeros((n_test_samples,self.n_classes))
            for i in range(self.n_estimators):
                predictions[np.arange(n_test_samples),self.trees[i].predict(test_images).astype(np.int32)]+=1
            return predictions/self.n_estimators
        else:
            predictions = np.zeros((self.n_estimators,n_test_samples,self.n_classes))
            for i in range(self.n_estimators):
                predictions[i] = self.trees[i].predict(test_images)
            return np.mean(predictions,axis=0)

    def predict(self,test_images):
        proba = self.predict_proba(test_images)
        return np.argmax(proba,axis=1,keepdims=True)
        pass

    def score(self,test_images,test_labels):
        # return : accuracy
        n_test = test_images.shape[0]
        test_images = test_images.reshape(n_test,-1)
        test_labels = test_labels.reshape(n_test,-1)
        pred_labels = self.predict(test_images).reshape(n_test,-1)
        acc = np.mean(pred_labels==test_labels)

        # info = "RandomForest score(): Sample number: {}, Accuracy: {}\n".format(test_labels.shape[0],acc)
        # logging.info(info)
        # if self.verbose:
        #     print(info)
        return acc

    @staticmethod
    def make_bdt(config,model_config,tree_config,i,__train_images,__train_labels,n_tree_samples):
        ## Design for parallel 
        start_time = time.time()
        if config.verbose and i%math.ceil(config.n_estimators/5)==0:
            info = f"Creating the {i+1}/{config.n_estimators} Base Decision Tree"
            print(info)
            logging.info(info)

        if config.bootstrap:
            train_images,train_labels = resample(__train_images ,__train_labels,n_tree_samples,tree_config["rdm"])
        else:
            idx = np.arange(len(__train_labels))
            np.random.shuffle(idx)  
            train_images,train_labels = __train_images[idx] ,__train_labels[idx]

        tree = BaseDecisionTree(config,model_config,tree_config)
        tree.fit(train_images,train_labels)

        if config.verbose and i%math.ceil(config.n_estimators/5)==0:
                info = f"Finish Creating the {i+1}/{config.n_estimators} Base Decision Tree"+ \
                    f" with depth {tree._get_tree_depth()}\n"
                info += f" time per tree: {round((time.time()-start_time),3)} s"
                print(info)
                logging.info(info)
        return tree

