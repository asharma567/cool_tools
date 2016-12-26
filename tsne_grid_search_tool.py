from sklearn.manifold import TSNE
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd

'''
todos
----
- use multithread so that each perplexity setting could optimized on a single thread
'''

grid_search_params = {
    
    # inits with PCA gives a better global structure
    'init': ['pca', 'random'],
    
    # 'precomputed' is also an option or 
    # also passing in a custom dist function
    'metric': ['euclidean'],
    
    #increases/decreases accuracy for the Barnes hut algo 
    'angle': [0.2, 0.5, 0.8],
    
    #defaults at 1000 but 5000 is known to work the best
    'n_iter': [3000, 5000],
    'learning_rate': [100, 500, 1000],
    'early_exaggeration': [2.0, 4.0, 6.0],
    
    #this actually affects the inputs to the cost-function
    'perplexity': range(5,50,10),   
}

def greedily_search_for_lowest_KL(perplexity_param, feature_M):
    best_kl_error = None
    best_params = None
    current_params = {}
    current_params['perplexity'] = perplexity_param
    current_params['init'] = 'pca'
    
    #find the lowest KL 
    for n_iter in grid_search_params['n_iter']:
        current_params['n_iter'] = n_iter
        
        #walk through all params related to tuning the cost-function            
        for learning_rate in grid_search_params['learning_rate']:
            current_params['learning_rate'] = learning_rate

            for early_exaggeration in grid_search_params['early_exaggeration']:
                current_params['early_exaggeration'] = early_exaggeration

                for angle in grid_search_params['angle']:
                    current_params['angle'] = angle
                

                    #fit tsne and return the embeddings & kl error
                    fitted_tsne = TSNE(**current_params).fit(feature_M)
                    embeddings, curr_error = fitted_tsne.embedding_, fitted_tsne.kl_divergence_
                    print curr_error
            
                    #greedy search tool
                    if not best_kl_error or best_kl_error > curr_error:
                        best_kl_error = curr_error
                        best_params = current_params
                        best_embeddings = embeddings
                    
    return best_params, best_embeddings, best_kl_error

def find_best_tsne_plot(feature_M, verbose=None):
    
    '''
    For every perplexity level we should optimize for the hyperparameters 
    for the lowest KL-Divergence
    '''
    
    #use multithreading to speed-up
    for perp in grid_search_params['perplexity']:
        print perp
        
        #find 
        best_params, embeddings, error = greedily_search_for_lowest_KL(perp, feature_M)
        
        #print plot for every configuration
        print best_params, perp, error
        plt.scatter(embeddings[:,0], embeddings[:,1], alpha=.2, color='red')
        plt.show()
    
    return best_params, embeddings


if __name__ == '__main__':
    s_curve, classes = datasets.make_s_curve(n_samples=300, noise=0.0)
    best_params, best_plot = find_best_tsne_plot(s_curve, verbose=True)
    print best_params