# Random Forest Model
# Hard Predict, No Soft Predict
import numpy as np
from scipy.linalg import qr
import math
from joblib import Parallel, delayed
# from utils import *
from sys import getsizeof
import gc
from model import *
# from regCCA_alt import *

class CCA(): 
    # Canonical Correlation Analysis
    def __init__(self,n_components=None):
        self.n = n_components
        self.is_fit = False
        self.A = None
        self.B = None

    def fit(self,X,Y):
        gammaX,gammaY,corrTol = 1e-9,1e-9,1e-10
        D,K = X.shape[1],Y.shape[1]

        XY = np.concatenate((X, Y), axis=1)
        C = np.cov(XY, rowvar=False)  ## [D+K, D+K]

        Cxx = C[0:D, 0:D] + (gammaX * np.eye(D))
        Cyy = C[D:, D:] + (gammaY * np.eye(K))
        Cxy = C[0:D, D:] ## [D, K]
        Cxx = (Cxx + Cxx.T)/2  ## make sure Cxx is symmetric, avoid numerical issues
        Cyy = (Cyy + Cyy.T)/2

        CholCxx = np.linalg.cholesky(Cxx).T
        if CholCxx.shape[0] == CholCxx.shape[1]: ## Compute inverse or pseudo-inverse
            invCholCxx = np.linalg.solve(CholCxx, np.eye(D))
        else:
            invCholCxx = np.linalg.lstsq(CholCxx, np.eye(D))

        CholCyy = np.linalg.cholesky(Cyy).T
        if CholCyy.shape[0] == CholCyy.shape[1]:
            invCholCyy = np.linalg.solve(CholCyy, np.eye(K))
        else:
            invCholCyy = np.linalg.lstsq(CholCyy, np.eye(K))

        T = invCholCxx.T.dot(Cxy).dot(invCholCyy) # [D, K]

        if D >= K: ## Choose the smaller one to do SVD, faster
            [L,S,D] = np.linalg.svd(T, full_matrices=False) 
            #  S: [K,], only diagonal elements; descending order 
            D = D.T
            A = invCholCxx.dot(L)
            B = invCholCyy.dot(D)
        else:
            [L,S,D] = np.linalg.svd(T.T, full_matrices=False)
            D = D.T
            A = invCholCxx.dot(D)
            B = invCholCyy.dot(L)

        r = S
        col_tol = np.absolute(r) > np.absolute(corrTol * np.max(np.absolute(r)))

        A = A[:, col_tol]
        B = B[:, col_tol]
        r = r[col_tol]

        if self.n is None:
            self.n = A.shape[1]      
        self.A = A[:,:self.n]
        self.B = B[:,:self.n]
        self.is_fit = True
        return self
    
    def __fit(self,X,Y):
        from sklearn.cross_decomposition import CCA as sk_CCA 
        self.n = self.n if self.n is not None else min(np.linalg.matrix_rank(X),np.linalg.matrix_rank(Y))

        skcca = sk_CCA(n_components=self.n )
        skcca.fit(X,Y)
        self.A = skcca.x_weights_
        self.B = skcca.y_weights_
        self.is_fit = True
        return self

    def _fit(self,X,Y):
        corrTol = 1e-10
        D = X.shape[1]
        K = Y.shape[1]
        X = (X - X.mean(axis=0))/(X.std(axis=0)+1e-30)
        Y = (Y - Y.mean(axis=0))/(Y.std(axis=0)+1e-30)
        ## QR Decomposition
        # Qx,Rx = np.linalg.qr(X,pivot=True)
        Qx,Rx,Px = qr(X,pivoting=True)
        Qy,Ry,Py = qr(Y,pivoting=True)
        __ix = np.absolute(np.diag(Rx))<np.absolute(Rx[0,0])*corrTol
        ni_X = np.max(np.where(__ix))if np.any(__ix) else len(__ix)
        __iy = np.absolute(np.diag(Ry))<np.absolute(Ry[0,0])*corrTol
        ni_Y = np.max(np.where(__iy))if np.any(__iy) else len(__iy)
        Qx = Qx[:,:ni_X]
        Qy = Qy[:,:ni_Y]
        Rx = Rx[:ni_X,:ni_X]
        Ry = Ry[:ni_Y,:ni_Y]
        v = min(ni_X,ni_Y)
        ## SVD
        if ni_X > ni_Y:
            U,S,V = np.linalg.svd(Qx.T@Qy)
        else:
            V,S,U = np.linalg.svd(Qy.T@Qx)
        U = U[:,:v]
        V = V[:,:v]
        _A = np.linalg.lstsq(Rx,U,rcond=0)[0]
        _B = np.linalg.lstsq(Ry,V,rcond=0)[0]
        # print(_A.shape,_B.shape,Px.shape,Py.shape,v,D,K,ni_X,ni_Y,__ix,__iy,Px,Py)
        A = np.zeros((D,v))
        B = np.zeros((K,v))
        A[Px,:] = np.concatenate((_A.T,np.zeros((v,D-ni_X))),axis=1).T if D-ni_X>0 else _A
        B[Py,:] = np.concatenate((_B.T,np.zeros((v,K-ni_Y))),axis=1).T if K-ni_Y>0 else _B

        if self.n is None:
            self.n = A.shape[1]      
        self.A = A[:,:self.n]
        self.B = B[:,:self.n]
        self.is_fit = True
        return self

    def transform(self,X,Y):
        if not self.is_fit:
            self.fit(X,Y)
        X_ = X.dot(self.A)
        Y_ = Y.dot(self.B)
        return X_,Y_
    
    def transform_X(self,X):
        if not self.is_fit:
            raise NotImplementedError
        X_ = X.dot(self.A)
        return X_

    def corrcoef(self,X,Y):
        X_,Y_ = self.transform(X,Y)
        return np.corrcoef(X_,Y_)[0,1]


class CCF_TreeNode(TreeNode): 
    def __init__(self,config,model_config,tree_config,depth=0,features_set=None):
        super().__init__(config,model_config,tree_config,depth,features_set)
        self.projection_bootstrap = self.config.projection_bootstrap
        self.a = None

    def fit(self,train_images,train_labels):
        self.n_node_samples = train_images.shape[0]
        ## Sort Data

        # Exclude Edge Cases
        if (self.model_config["max_depth"] is not None and 
            self.depth >= self.model_config["max_depth"]) or \
            self.n_node_samples < self.model_config["min_samples_split"] or \
            self.features_set == None or \
            np.unique(train_images[:,self.features_set],axis=0).shape[0] == 1:

            self.be_leaf(train_labels)
            return self
        if len(np.unique(train_labels)) == 1:
            self.be_leaf(train_labels)
            return self

        # Choose split feature and split value
        if self.model_config["max_features"]==None:
            self.m_features = int(np.sqrt(self.tree_config["n_features"]))
        else:
            self.m_features = int(self.model_config["max_features"]*self.tree_config["n_features"])
        
        
        choices = np.random.choice(self.features_set,self.m_features,replace=False)
        self.choices = choices
        if self.projection_bootstrap:
            X_dot,Y_dot = resample(train_images[:,choices],train_labels,self.n_node_samples,self.tree_config["rdm"]) 
        else:
            X_dot,Y_dot = train_images[:,choices],train_labels 
        # Y: one hot
        Y_dot = np.eye(self.tree_config["n_classes"])[Y_dot.astype(np.int32).reshape(-1)] 

        a_CCA = CCA(n_components=None)
        a_CCA.fit(X_dot,Y_dot)
        v = a_CCA.n
        if v==0:
            self.be_leaf(train_labels)
            return self

        U = a_CCA.transform_X(train_images[:,choices])
        self.gain_base = self._criterion(train_labels)
        self.split_feature = self.choose_split_feature_ccf(U,train_labels,v)
        sort_ind = np.argsort(U[:,self.split_feature])
        train_images = train_images[sort_ind]
        U = U[sort_ind]
        train_labels = train_labels[sort_ind]
        self.a = a_CCA.A[:,self.split_feature]

        del a_CCA,sort_ind,X_dot,Y_dot,v
        gc.collect()
        self.split_value,self.split_gain = \
            self.choose_split_value(U,train_labels,self.split_feature)
        self.real_gain = (self.split_gain)\
                            *self.n_node_samples/self.tree_config["n_tree_samples"] 

        assert self.a.shape[0] == self.m_features, \
            f"self.a.shape[0] should be equal to self.m_features, {self.a.shape[0], self.m_features}"
        # Split
        _,left_labels,_,right_labels,left_index,right_index = \
            self.split(feature=self.split_feature,split_value=self.split_value,\
                        train_images=U,train_labels=train_labels)
        del U,_
        # print("DEBUG:",gc.collect(),[ (k,getsizeof(v),id(v)) for k,v in locals().items()])
        # Exclude Edge Cases
        if left_labels.shape[0] <= self.tree_config["min_samples_leaf"] or \
            right_labels.shape[0] <= self.tree_config["min_samples_leaf"] or \
            self.real_gain  < self.model_config["min_impurity_decrease"]:

            self.be_leaf(train_labels)
            return self
        
        # Build Child
        self.child.append(CCF_TreeNode(self.config,self.model_config,\
                                   self.tree_config,self.depth+1,self.features_set))
        self.child.append(CCF_TreeNode(self.config,self.model_config,\
                                   self.tree_config,self.depth+1,self.features_set))
        left_images = train_images[left_index]
        right_images = train_images[right_index]
        self.child[0].fit(left_images,left_labels)
        self.child[1].fit(right_images,right_labels)
        # self.child[0].fit(train_images[left_index],left_labels)
        # self.child[1].fit(train_images[right_index],right_labels)
        # plot_test(train_images,train_labels,clf=self)

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
            if self.a is not None:
                test_U = np.matmul(test_images[:,self.choices],self.a).reshape(-1,1)
            else:
                raise NotImplementedError
            
            left_images,left_labels,right_images,right_labels,left_index,right_index = \
                self.split(0,self.split_value,test_U,None)
            left_pred = self.child[0].predict(test_images[left_index])
            right_pred = self.child[1].predict(test_images[right_index])
            if not self.config.soft_pred:
                pred = np.zeros(test_images.shape[0])
                pred[left_index] = left_pred
                pred[right_index] = right_pred
            else:
                pred = np.zeros((test_images.shape[0],self.tree_config["n_classes"]))
                pred[left_index] = left_pred
                pred[right_index] = right_pred
            return pred


    def choose_split_feature_ccf(self,train_images,train_labels,v):
        choices = np.arange(v)
        best_feature = None
        best_gain = -np.inf
        criterion = self.model_config["criterion"]
        for feature in choices:
            gain = self.get_split_feature_gain(train_images,train_labels,feature,criterion)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        return best_feature
    

class CCF_BDT(BaseDecisionTree): 
    def __init__(self,config,model_config,tree_config):
        super().__init__(config,model_config,tree_config)

    def fit(self,train_images,train_labels):
        self.tree_config["n_tree_samples"] = train_images.shape[0]
        self.root = CCF_TreeNode(self.config,self.model_config,self.tree_config)
        self.root.fit(train_images,train_labels)
        return self

class CCF(RandomForest):
    def __init__(self,config,model_config):
        super().__init__(config,model_config)

    def fit(self,train_images,train_labels):
        start_time = time.time()

        # 0 mean 

        __train_images = train_images.reshape(train_images.shape[0],-1)
        __train_labels = train_labels.reshape(train_labels.shape[0],-1)
        self.train_images_mean = __train_images.mean(axis=0)
        __train_images = __train_images - self.train_images_mean
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
            info = "Creating the CCF!\n"
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


        if self.verbose:
            info = f"Finish Creating the CCF! with {self.n_estimators} Base Decision Trees"+\
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
        test_images = test_images - self.train_images_mean
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

    @staticmethod
    def make_bdt(config,model_config,tree_config,i,__train_images,__train_labels,n_tree_samples):
        ## Design for parallel 
        start_time = time.time()
        if config.verbose and i%math.ceil(config.n_estimators/5)==0:
            info = f"Creating the {i+1}/{config.n_estimators} CCF Base Decision Tree"
            print(info)
            logging.info(info)

        if config.bootstrap:
            train_images,train_labels = resample(__train_images ,__train_labels,n_tree_samples,tree_config["rdm"])
        else:
            idx = np.arange(len(__train_labels))
            np.random.shuffle(idx)  
            train_images,train_labels = __train_images[idx] ,__train_labels[idx]


        tree = CCF_BDT(config,model_config,tree_config)
        tree.fit(train_images,train_labels)

        if config.verbose and i%math.ceil(config.n_estimators/5)==0:
                info = f"Finish Creating the {i+1}/{config.n_estimators} CCF Base Decision Tree"+ \
                    f" with depth {tree._get_tree_depth()}\n"
                info += f" time per tree: {round((time.time()-start_time),3)} s"
                print(info)
                logging.info(info)
        return tree

