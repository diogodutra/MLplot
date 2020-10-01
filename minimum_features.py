import numpy as np
import pandas as pd


def intersection_pair(hist_1, hist_2):
    """ Calculate the area of the intersection between a pair of histograms."""
    
    minima = np.minimum(hist_1, hist_2)
    intersection_i = np.true_divide(np.sum(minima), np.sum(hist_1))
    intersection_j = np.true_divide(np.sum(minima), np.sum(hist_2))
    
    return intersection_i, intersection_j


def intersection_array(list_hist):
    """ Calculate the normalized area of each histogram out of the intersections."""
    
    n_hist = len(list_hist)
    I = np.zeros(n_hist*(n_hist-1))
    k = 0
    for i in range(n_hist):
        for j in range(i+1, n_hist):
            I[k], I[k+1] = intersection_pair(list_hist[i], list_hist[j])
            k += 2
            
    return I


def intersections_per_features(df, labels, bins=5):
    features = df.columns
    n_features = len(features)
    classifications = list(set(labels))
    n_classifications = len(classifications)
    n_classifications_pairs = n_classifications*(n_classifications-1)

    I = np.zeros((n_features, n_classifications_pairs))
    for f, feature in enumerate(features):

        hist = []

        val_min = min(df[feature])
        val_max = max(df[feature])

        # calculate histograms for each type of target
        for i, classification in enumerate(classifications):
            filter_class = [classification==y for y in labels] 
            hist_i, _ = np.histogram(df[filter_class][feature], bins=bins, range=[val_min, val_max])
            hist.append(hist_i)

        I[f,:] = intersection_array(hist)


    cols = []
    for i in range(n_classifications):
        for j in range(i+1, n_classifications):
            cols.append(str(classifications[i]) + '_|_' + str(classifications[j]))
            cols.append(str(classifications[j]) + '_|_' + str(classifications[i]))

    I = pd.DataFrame(data=I, index=features, columns=cols)
        
    return I

def minimum_features(df, labels, step_threshold=.1):
    """
    Return a subset of features that creates the best
    contrast between the discrete classifications.

    Assumptions:
     - all columns (except target) are features.
     - all features are continuous.
     - all classes (in target column) are discrete labels.
     - many samples in order to spot intersections between discrete probability distributions.
     """

    I = intersections_per_features(df, labels)

    n_cols = len(I.columns)
    n_features = len(I.index)
    intersections_unresolved = np.array([True]*n_cols)
    index_sort = []
    is_all_intersections_separated = False
    is_all_features_sorted = False

    for threshold in np.arange(0.0, 1.0+step_threshold, step_threshold):
        for c in range(len(index_sort), n_cols):

            I_unresolved = I.loc[:, intersections_unresolved]

            low_intersections = I_unresolved <= threshold
            count_low_intersections = I_unresolved[low_intersections].count(axis=1)
            is_there_low_intersections = count_low_intersections.max() > 0

            if is_there_low_intersections:
                best_new_feature = count_low_intersections.idxmax()
                if best_new_feature not in index_sort:
                    index_sort.append(count_low_intersections.idxmax())

                intersections_unresolved = (I.loc[index_sort,:] > threshold).max(axis=0).values

            else:
                break

            is_all_intersections_separated = sum(intersections_unresolved) == 0
            is_all_features_sorted = len(index_sort) == n_features

            if is_all_intersections_separated or is_all_features_sorted:
                break

        if is_all_intersections_separated or is_all_features_sorted:
            break


    # calculate separations by index_sort (best_features) for each classification
    I_min = I.loc[index_sort, :].min(axis=0)
    classifications = list(set(labels))
    n_classifications = len(classifications)
    S = np.zeros((n_classifications, n_classifications))
    for i, i_class in enumerate(classifications):
        for j, j_class in enumerate(classifications):
            if i!=j:
                S[i,j] = I_min[str(i_class)+'_|_'+str(j_class)]

    intersections = S.max(axis=1)
    intersections = pd.DataFrame(data=intersections,
                                 columns=['intersections'],
                                 index=classifications
                                ).transpose()

    return index_sort, intersections, I