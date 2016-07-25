def multithread_map(fn, work_list, num_workers=50):
    from concurrent.futures import ThreadPoolExecutor
    '''
    spawns a threadpool and assigns num_workers to some 
    list, array, or any other container. Motivation behind 
    this was for functions that involve scraping.
    '''
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(fn, work_list))


#FIX this
def get_scores(args):
    
    #done this way to support the map function within the threadpool
    clf, x_train, encoded_y_train = args 
    
    #this is done to hold the training/tuning fixed
    kf = KFold(len(x_train), n_folds=10, random_state=42)

    #grabbing scores
    scores = cross_val_score(
                            clf, 
                            x_train, 
                            encoded_y_train, 
                            cv=kf, 
                            
                            #put in arg to change this
                            scoring=f1_score
                    )
    return np.mean(scores), x_train.shape[0]

def get_optimal_number_of_training_examples(clf, x_train, y_train, number_of_steps=100, step_size=100, multithread=False):

    if multithread:
        #rename i to curr_step
        work_list = [(clf, x_train[step_size * i:], encoded_y_train[step_size * i:]) for i in range(number_of_steps)]
        score_keeper = multithread_map(get_scores, work_list)
    else:
        #simple way to do it
        score_keeper = []
        for i in range(number_of_steps):
            number_of_examples_to_trim_off = step_size * i
            score_keeper.append(
                get_scores((
                    clf, x_train[number_of_examples_to_trim_off:], 
                    encoded_y_train[number_of_examples_to_trim_off:]
                ))
            )

    
    df = pd.DataFrame(score_keeper).set_index(1)
    df.sort_index().rename(columns={0:'performance', 1:'number_of_examples'}).plot()
