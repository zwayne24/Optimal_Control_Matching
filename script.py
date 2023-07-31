#####################################################################################################################
#                                               PACKAGE INSTALLATION
#####################################################################################################################

import sys 
import os 
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LassoCV, ElasticNetCV, RidgeCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, Binarizer,  OneHotEncoder 
from factor_analyzer import FactorAnalyzer
import inquirer
import pulp 
from scipy.stats import pearsonr
import time
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#####################################################################################################################
#                                                 FUNCTIONS
#####################################################################################################################

######################################################################
# 1. Reading and transforming data
######################################################################
# Functions: read_data, transform_data, get_original_data

def read_data(path='../Data/Final/CLEAN_DATA_22_Imputed.csv', outcome='STAAR_Meets', treatment='SURVEY_CRIMSI'):
    """
    Read in data and split into covariates, treatment, and outcome

    Parameters:
    ----------
    path : str
        Path to data.
    outcome : str
        Name of outcome variable.
    treatment : str
        Name of treatment variable.
    
    Returns:
    -------
    X : pd.DataFrame
        Covariates.
    T : pd.Series
        Treatment.
    y : pd.Series
        Outcome.

    """
    data = pd.read_csv(path)  #load data
    y = data[outcome] #define outcome variable
    T = data[treatment] #define treatment variable
    X = data.drop([outcome, treatment], axis=1) #define covariatest
    return X, T, y

def transform_data(X):
    """
    Transform data by standardizing numerical variables, binarizing binary variables, and one-hot encoding categorical variables.

    Parameters:
    ----------
    X : pd.DataFrame
        The feature matrix.
    
    Returns:
    -------
    X_trans: pd.DataFrame
        The standardized feature matrix.
    """

    bin_cols = [col for col in X.columns if set(X[col]) == {0, 1}]
    cat_cols = [col for col in X.columns if X[col].dtype == 'object' or X[col].nunique() < 8 and col not in bin_cols]
    num_cols = [col for col in X.columns if col not in bin_cols + cat_cols]

    # Define the column transformer to standardize numerical and binary variables separately
    ct = ColumnTransformer([ ('num', StandardScaler(), num_cols), ('bin', Binarizer(), bin_cols), ('cat', OneHotEncoder(drop='first'), cat_cols)])

    # Fit the column transformer to the training data
    ct.fit(X)

    # Get the feature names after transformation
    feature_names = num_cols + bin_cols + list(ct.named_transformers_['cat'].get_feature_names_out(cat_cols))

    # Transform the data and create a new DataFrame with the transformed features
    X_trans = pd.DataFrame(ct.transform(X), columns=feature_names)

    return X_trans


def get_original_data(matches, ogdata):
    '''
    This function takes in a dataframe with matched units and returns a dataframe with the original data for the matched units.
    
    Parameters:
    -----------
    matches : dataframe
        Dataframe with matched units.
    ogdata : dataframe
        Dataframe with original data.
    
    Returns:
    --------
    ogmatches : dataframe
        Dataframe with original data for the matched units.
    '''
    # make a new dataframe with the same columns as the original data
    ogmatches = pd.DataFrame(columns=ogdata.columns)
    
    # iterate through the matches dataframe and append the original data for each match based on the index
    for i, _ in matches.iterrows():
        ogmatches = pd.concat([ogmatches, ogdata.loc[[i]]])
    
    # add the match number and treatment columns to the dataframe
    ogmatches['match_num'] = matches['match_num']
    ogmatches['treatment'] = matches['treatment']
    
    # add the propensity score column if it exists
    if 'propensity_score' in matches.columns:
        ogmatches['propensity_score'] = matches['propensity_score']
    
    return ogmatches

####################################################################################
# 2. Propensity score methods - factor analysis, getting scores
####################################################################################
# Functions: factor_analysis, logistic_regression, ps_balanced

def factor_analysis(X, T, k=None):
    """
    Reduce the dimensionality of X using Factor Analysis to k number of factors.

    Parameters:
    ----------
    X : pd.DataFrame
        The feature matrix.
    T : pd.Series
        The treatment vector.
    k : int
        The number of factors to keep.

    Returns:
    -------
    X_factor : pd.DataFrame
        The factor matrix.
    fa : FactorAnalyzer
        The fitted FactorAnalyzer object.
    """

    # if k is not given, define k with the 1:10 rule, where k is the number of factors
    if k is None:
        k = int(np.floor(X[T==1].shape[0]/10))
    
    fa = FactorAnalyzer(n_factors=k, rotation='varimax')
    fa.fit(X)
    X_factor = fa.transform(X)
    X_factor = pd.DataFrame(X_factor).rename(columns=lambda x: 'Factor_' + str(x+1))
    return X_factor, fa


def logistic_regression(X, T):
    """
    Compute propensity scores using logistic regression. Propensity scores are the probability of each row to belong to the treatment group.

    Parameters:
    ----------
    X : pd.DataFrame
        The feature matrix.
    T : pd.Series
        The treatment vector.
    
    Returns:
    -------
    X : pd.DataFrame
        The feature matrix with propensity scores.

    """

    X = X.copy()
    # fit logistic regression model
    log_reg = LogisticRegression(max_iter=1000, random_state=0) #
    log_reg.fit(X, T)
    # predict probabilities for each row to belong to treatment group
    probs = log_reg.predict_proba(X)[:,1] 
    X['propensity_score'] = probs
    X['treatment'] = T
    return X


def ps_balanced(X, T):
    """
    Repeatedly compute propensity scores with random subsamples of the control group to solve the class imbalance problem between treated and control.
    We repeat this process until all control units have a propensity score, and we average all the computed scores for the treated units to get the final propensity scores.

    Parameters:
    ----------
    X : pd.DataFrame
        The feature matrix.
    T : pd.Series
        The treatment vector.
    
    Returns:
    -------
    ps : pd.DataFrame
        The propensity scores.
    """

    # Create a dataframe to store the propensity scores with shape of X
    ps = pd.DataFrame(index=X.index)
    
    # Define control and treated groups
    control = X[T == 0]
    treated = X[T == 1]
    
    # Determine the number of subsamples
    n_subsamples = int(np.ceil(control.shape[0] / treated.shape[0]))
    
    # Determine the number of control units per subsample
    control_per_subsample = treated.shape[0]
    
    # Randomly shuffle the control units
    shuffled_control = control.sample(frac=1, random_state=42)
    
    # Create a list to store the propensity score columns
    propensity_score_cols = []
    
    # Create a for loop to compute propensity scores for each subsample
    for i in range(n_subsamples):
        # Calculate the start and end indices for the current subsample
        start_idx = i * control_per_subsample
        end_idx = min((i + 1) * control_per_subsample, control.shape[0])
        
        # Get the control units for the current subsample
        control_sample = shuffled_control.iloc[start_idx:end_idx]
        
        # Combine the control sample and the treated group
        sample = pd.concat([control_sample, treated], axis=0)
        
        # Compute propensity scores for the sample
        sample_ps = logistic_regression(sample, T.loc[sample.index])
        
        # Append the propensity scores to the list
        propensity_score_cols.append(sample_ps['propensity_score'])
    
    # Concatenate all the propensity score columns into the final dataframe
    ps = pd.concat(propensity_score_cols, axis=1)
    ps.columns = [f'propensity_score_{i}' for i in range(n_subsamples)]
    
    # Compute the final propensity score as the mean of the subsample propensity scores 
    ps['FINAL_PS'] = ps.mean(axis=1) 
    return ps

####################################################################################
# 3. Feature selection
####################################################################################
# Functions: feature_selection

def feature_selection(X, y, cv=5, method='lasso', num_selected=15, max_correlation=0.6, must_include=None):
    """
    Performs feature selection using Lasso, Ridge, or ElasticNet with optimization and adding multicollinearity and sparsity constraints.

    Parameters:
    ----------
    X : pd.DataFrame
        The feature matrix.
    y : pd.Series
        The outcome vector.
    cv : int
        Number of folds for cross-validation.
    method : str
        The method to use for feature selection. Must be 'lasso', 'ridge', or 'elastic'.
    num_selected : int
        The number of features to select.
    max_correlation : float
        The maximum correlation allowed between two features.
    must_include : list, optional
        The list of features that must be included in the matching model.
    
    Returns:
    -------
    selected_features : list
        The list of selected features names.
    coef_all : list
        The list of coefficients of all features.
    coef_selected : list
        The list of coefficients of selected features.

    """
    num_features = X.shape[1]

    # Define the model
    if method == 'lasso':
        model = LassoCV(cv=cv)
    elif method == 'ridge':
        model = RidgeCV(cv=cv)
    elif method == 'elastic':
        model = ElasticNetCV(cv=cv)
    else:
        raise ValueError('Method must be "lasso", "ridge", or "elastic"')

    # Fit the model to the data
    model.fit(X, y)

    # Get the optimal value of alpha and the corresponding coefficients
    alpha = model.alpha_
    coef = model.coef_

    # Create the optimization problem
    prob = pulp.LpProblem("Feature_Selection", pulp.LpMaximize)

    # Create binary decision variables for each feature
    selected = pulp.LpVariable.dicts("selected", range(num_features), cat="Binary")

    # Set the objective function: maximize the sum of absolute selected coefficients
    prob += pulp.lpSum([selected[i] * abs(coef[i]) for i in range(num_features)])

    # Add constraints
    # Constraint 1: Avoid multicollinearity (correlation threshold)
    for i in range(len(X.columns)):
        for j in range(i + 1, len(X.columns)):
            if abs(pearsonr(X.iloc[:, i], X.iloc[:, j])[0]) > max_correlation:
                prob += selected[i] + selected[j] <= 1

    # Constraint 2: Limit the number of selected features (sparsity constraint)
    prob += pulp.lpSum([selected[i] for i in range(len(X.columns))]) <= num_selected

    # Constraint 3: Must include certain features
    if must_include is not None:
        for feature in must_include:
            # Get the index of the feature and set x to 1 (selected)
            feature_index = X.columns.tolist().index(feature)
            prob += selected[feature_index] == 1


    # Solve the problem and don't show the output
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Get the selected feature indices
    selected_indices = [i for i in range(num_features) if selected[i].varValue == 1]

    # Get the selected feature names
    selected_features = X.columns[selected_indices].tolist()

    # Save coefficients of all features in a list
    coef_all = coef.tolist()

    # Save coefficients of selected features in a list
    coef_selected = [coef[i] for i in selected_indices]

    return selected_features, coef_all, coef_selected

####################################################################################
# 4. Distance matrix 
####################################################################################
# Functions: ps_distance_matrix, distance_matrix_hard_constraints

def ps_distance_matrix(data, T, PS, hc):
    """
    Calculate the weighted (euclidean) distance matrix between treated and control units for the features specified.

    Parameters:
    -----------
    T : pd.series
        Series with treatment indicator. 1 if treatment, 0 if control.
    PS : pd.series
        Series with propensity scores.
    
    Returns:
    --------
    D : pandas.DataFrame
        The distance matrix of propensity scores where rows correspond to treated units and columns correspond to control units. 
        The row and column names are the indexes of the original data.

    """
    data = data.copy()
    data['PS'] = PS
    
    # Define treated and control units
    treated = data[T == 1]
    control = data[T == 0]

    # If no features are specified, use all features
        
    features = ['PS']

    # Check if hard constraint features are present in the list of features
    if hc is not None:
        missing_features = set(hc) - set(features)
        if missing_features:
            # Add missing hard constraint features to the list of features
            features += missing_features

    # Calculate the distance between each treated and control unit pair based on distance between each feature
    treated_features = treated[features].values
    control_features = control[features].values

    # Calculate the distances for all features
    distances = np.abs((treated_features[:, np.newaxis] - control_features))

    # Apply penalty to features with hard constraints
    if hc is not None:
        hard_constraint_indices = [features.index(feature) for feature in hc]
        distances[:, :, hard_constraint_indices] *= 1e9
    
    # Sum the distances across features
    D = np.sum(distances, axis=2)

    # Transform into a DataFrame and add row and column names as indexes of the original data
    D = pd.DataFrame(D, index=treated.index, columns=control.index)

    return D


def distance_matrix_hard_constraints(X, T, features=None, hard_constraint_features=None, weights=None, penalty=1e9):
    """
    Calculate the weighted (absolute euclidean) distance matrix between treated and control units for the features specified,
    adding penalty for hard constraints features that don't have the same value between treated and control units.

    Parameters:
    -----------
    X : pandas.DataFrame
        The feature matrix.
    T : pd.series
        Series with treatment indicator. 1 if treatment, 0 if control.
    features : list, optional
        The list of features to be used for distance calculation (default: None). If None, all features are used.
    weights : list, optional
        The list of weights to be used for distance calculation (default: None). If None, features are not weighted.
    hard_constraint_features : list, optional
        The list of features to be used for distance calculation with hard constraints (default: None). If None, no features have hard constraints.
    weights: list, optional
        The list of weights to be used for distance calculation (default: None). If None, features are not weighted.
    penalty : float, optional
        The penalty to be applied to the distance between treated and control units for features with hard constraints. If penalty is higher, we ensure features with hard constraints are closer.
    
    Returns:
    --------
    D : pandas.DataFrame
        The distance matrix where rows correspond to treated units and columns correspond to control units.
        The row and column names are the indexes of the original data.
    
    """
    
    # Define treated and control units
    treated = X[T == 1]
    control = X[T == 0]

    # Define subset of features to be used for distance calculation
    if features is None: # If no features are specified, use all features
        features = X.columns.tolist()
    else:
        features = features.copy() # Make a copy of the list of features to avoid modifying the original list

    # Check if hard constraint features are present in the list of features
    if hard_constraint_features is not None:
        missing_features = set(hard_constraint_features) - set(features) # Get the list of hard constraint features that are not in the list of features
        if missing_features: 
            features += missing_features # Add missing hard constraint features to the list of features
            if weights is not None:
                weights+= [1]*len(missing_features) # Add weights for the missing hard constraint features
    
    # Define weights for each feature
    if weights is None: # If no weights are specified, use equal weights for all features
        weights = np.ones(len(features))

    # Select the values of the relevant features from treated and control units
    treated_features = treated[features].values
    control_features = control[features].values

    # Calculate the abolsute distances for all features
    distances = np.abs((treated_features[:, np.newaxis, :] - control_features) * weights)

    # Apply penalty to features with hard constraints
    if hard_constraint_features is not None:
        hard_constraint_indices = [features.index(feature) for feature in hard_constraint_features] # Get the indices of the hard constraint features
        distances[:, :, hard_constraint_indices] *= penalty # Apply penalty to the distances of the hard constraint features
    
    # Sum the distances across features to get the final distance matrix
    D = np.sum(distances, axis=2)

    # Transform into a DataFrame and add row and column names as indexes of the original data
    D = pd.DataFrame(D, index=treated.index, columns=control.index)

    return D

#########################################################################################
# 5. Matching
#########################################################################################
# Functions: dist_matching, dist_matching_PS_calipers

def dist_matching(X, T, dist, with_replacement=True, optimal=False, num_matches=1):
    """
    This function takes in a dataframe with treatment and control units, a dataframe with distances between treatment and control units, and a matching type
    and returns a dataframe with matched units.

    Parameters:
    -----------
    X : pd.dataframe
        Feature matrix.
    T : pd.series
        Series with treatment indicator. 1 if treatment, 0 if control.
    dist : pd.dataframe
        Dataframe with distances between treatment (rows) and control units (columns).
    with_replacement : boolean
        Whether or not to match with replacement. 
    optimal : boolean
        Whether or not to use optimal matching.
    num_matches : int
        Number of control matches per treatment unit. If 1, one-to-one matching is performed. If >1, one-to-many matching is performed.
    
    Returns:
    --------
    match_full : dataframe
        Dataframe with matched units.
    """
    X['treatment'] = T

    # Identify treatment and control units
    treatment = X[T == 1]
    control = X[T == 0]

    matches = [] # Create an empty list to store matches

    # -------- OPTION 1: WITH REPLACEMENT ----------
    if with_replacement:
        match_num = 1 # Initialize the match number
        for t_idx, t_row in treatment.iterrows(): # Iterate over the treatment units
            match_indices = dist.loc[t_idx][control.index].nsmallest(num_matches).index.tolist() # Get the indices of the k closest control units
            matches.extend([[t_idx, c_idx, match_num] for c_idx in match_indices]) # Add the matches to the matches list
            match_num += 1 # Increment the match number
    
    # -------- OPTION 2: WITHOUT REPLACEMENT ----------
    elif not with_replacement:
        # --------- OPTION 2A: GREEDY MATCHING ----------
        if not optimal:
            match_num = 1 # Initialize the match number
            matched_controls = set()  # Keep track of matched control units
            for t_idx, t_row in treatment.iterrows():  # Iterate over the treatment units
                available_controls = [c_idx for c_idx in control.index if c_idx not in matched_controls]
                control_indices = dist.loc[t_idx][available_controls].nsmallest(num_matches).index.tolist()
                matches.extend([[t_idx, c_idx, match_num] for c_idx in control_indices])
                matched_controls.update(control_indices)
                match_num += 1 # Increment the match number

        # --------  OPTION 2B: OPTIMAL MATCHING (OPTIMIZATION) ----------
        elif optimal:
            print('Matching Process Ongoing')
            # Create optimization problem: minimization problem
            prob = pulp.LpProblem("Matching_Problem", pulp.LpMinimize)

            # Define decision variables: binary variable x_{ij} = 1 if treatment unit i is matched to control unit j, 0 otherwise
            x = pulp.LpVariable.dicts("x", [(t_idx, c_idx) for t_idx in treatment.index for c_idx in control.index], cat='Binary')

            # Objective function: minimize total distance
            prob += pulp.lpSum(dist.loc[t_idx][c_idx] * x[(t_idx, c_idx)] for t_idx in treatment.index for c_idx in control.index)

            # Constraint 1: each treatment unit is matched to at least the number of control specified
            for t_idx in treatment.index: 
                    prob += pulp.lpSum(x[(t_idx, c_idx)] for c_idx in control.index) >= num_matches  

            # Constraint 2: each control unit is matched to at most one treatment unit
            for c_idx in control.index:
                prob += pulp.lpSum(x[(t_idx, c_idx)] for t_idx in treatment.index) <= 1

            # Solve the optimization problem
            start_time = time.time()  # Record the start time
            prob.solve(pulp.GLPK_CMD(msg=0)) # Use GLPK solver and suppress output
            end_time = time.time()  # Record the end time
            execution_time = end_time - start_time # Calculate the execution time
            print("Execution Time:", execution_time, "seconds")

            # Retrieve the matches from the decision variables 
            match_num = 1  # Initialize the match number
            matched_treatment = set()  # Keep track of treatment units that have been matched
            for t_idx in treatment.index: # Iterate over the treatment units
                matched_control = []  # Keep track of control units matched with this specific treatment unit
                for c_idx in control.index: # Iterate over the control units
                    if pulp.value(x[(t_idx, c_idx)]) == 1: # If the decision variable is 1, the pair is a match
                        matched_control.append(c_idx) # Add the control unit index to the matched control list
                if matched_control: # If the matched control list is not empty
                    if t_idx not in matched_treatment: # If the treatment unit has not been matched yet, it is not in the set
                        for c_idx in matched_control: # Iterate over the matched control units to this specific treatment unit
                            matches.append([t_idx, c_idx, match_num]) # Add the matched pair to the matches list with the match number
                        matched_treatment.add(t_idx) # Add the treatment unit index to the matched treatment set to keep track of matched treatment units
                        match_num += 1 # Increment the match number
    
    # If the problem found matches then return the matches
    if matches:
        match_full = pd.DataFrame(columns=X.columns)  # Create an empty dataframe to store the matches including the features from X

        # Iterate over the matches and include the matched number column 
        for match in matches:
            match_treatment = treatment.loc[match[0]].to_frame().T # Get matched treatment unit features values from X
            match_control = control.loc[match[1]].to_frame().T # Get matched control unit features values from X
            match_treatment['match_num'] = match[2]  # Use the match number from the matches list
            match_control['match_num'] = match[2]  # Use the match number from the matches list
            match_full = pd.concat([match_full, match_treatment, match_control]) # Concatenate the matched treatment and control units to the match_full dataframe
                
        # Add a column for the treatment indicator from T
        match_full['treatment'] = T[match_full.index]

        #drop duplicated treatment rows that are added for when num_matches>1 (just for formatting purposes)
        match_full=match_full[~((match_full['treatment']==1) & (match_full['match_num'].duplicated()))]
        print('Matching Process Complete!')
        return match_full
    
    # If the problem did not find matches, then print a suggestion to relax the constraints
    elif not matches:
        print('** WARNING ** No matches found. You should try to relax your constraints in the optimization problem as the problem may be infeasible. Also, you can try to reduce the number of control matches as there might not be enough control units to match to.')
        return None
    

def dist_matching_PS_calipers(X, T, PS, dist, with_replacement=True, optimal=False, num_matches=1, caliper=.02,
                                 hard_constraints=None):
    """
    Matches treated and control units based on the distance matrix and propensity scores with calipers including hard constraints in the optimization.

    Parameters:
    -----------
    X : pandas.DataFrame
        The feature matrix.
    T : pandas.Series
        Series with treatment indicator. 1 if treatment, 0 if control.
    PS : pandas.Series
        Series with propensity scores.
    dist : pandas.DataFrame
        The distance matrix where rows correspond to treated units and columns correspond to control units.
    with_replacement : bool, optional
        Whether to match with replacement (default: True).
    optimal : bool, optional
        Whether to perform optimal matching (default: False). Creates optimal pairing when with_replacement = False.
    num_matches : int, optional
        Number of control matches per treatment unit. If 1, one-to-one matching is performed. If >1, one-to-many matching is performed.
    caliper : float, optional
        The caliper to be applied to the distance between treated and control units for the propensity score. If caliper is smaller, we ensure units matched have similar propensity scores.
    hard_constraints : list, optional
        The list of features to be used for distance calculation with hard constraints (default: None). If None, no features have hard constraints.
        
    Returns:
    --------
    match_full : pandas.DataFrame
        Dataframe with the matched units. The match_num column indicates the match number for each treated and control unit pair.

    """
    # Identify treatment and control units
    treatment = X[T == 1]
    control = X[T == 0]

    matches = [] # Create an empty list to store matches

    # -------- OPTION 1: WITH REPLACEMENT ----------
    if with_replacement:
        match_num = 1 # Initialize the match number
        for t_idx, t_row in treatment.iterrows(): # Iterate over the treatment units
            control_within_caliper = control[abs(PS.loc[t_idx] - PS.loc[control.index]) <= caliper] # Get the control units within the propensity score caliper
            match_indices = dist.loc[t_idx][control_within_caliper.index].nsmallest(num_matches).index.tolist() # Get the indices of the k closest control units within the caliper
            matches.extend([[t_idx, c_idx, match_num] for c_idx in match_indices]) # Add the matches to the matches list
            match_num += 1 # Increment the match number
        
    # -------- OPTION 2: WITHOUT REPLACEMENT ----------
    elif not with_replacement:
        # --------- OPTION 2A: GREEDY MATCHING ----------
        if not optimal:
            match_num = 1 # Initialize the match number
            matched_controls = set()  # Keep track of matched control units
            for t_idx, t_row in treatment.iterrows(): # Iterate over the treatment units
                available_controls = [c_idx for c_idx in control.index if c_idx not in matched_controls] # Safe the control units that have not been matched yet
                control_within_caliper_condition = abs(PS.loc[t_idx] - PS.loc[available_controls]) <= caliper # Boolean condition for the control units within the propensity score caliper
                control_within_caliper = [c_idx for c_idx, condition in zip(available_controls, control_within_caliper_condition) if condition] # Get the control indexes within the propensity score caliper
                control_indices = dist.loc[t_idx][control_within_caliper].nsmallest(num_matches).index.tolist() # Get the indices of the k closest control units within the caliper
                matches.extend([[t_idx, c_idx, match_num] for c_idx in control_indices]) # Add the matches to the matches list
                matched_controls.update(control_indices) # Add the matched control indices to the matched controls set to update the available controls
                match_num += 1 # Increment the match number
                
        # --------  OPTION 2B: OPTIMAL MATCHING (OPTIMIZATION) ----------
        if optimal:
            print('Matching Process Ongoing')
             # Create optimization problem: minimization problem
            prob = pulp.LpProblem("Matching_Problem", pulp.LpMinimize)

            # Define decision variables: binary variable x_{ij} = 1 if treatment unit i is matched to control unit j, 0 otherwise
            x = pulp.LpVariable.dicts("x", [(t_idx, c_idx) for t_idx in treatment.index for c_idx in control.index], cat='Binary')

            # Objective function: minimize total distance
            prob += pulp.lpSum(dist.loc[t_idx][c_idx] * x[(t_idx, c_idx)] for t_idx in treatment.index for c_idx in control.index)

            # Constraint 1: each treatment unit is matched to at least the number of control specified
            for t_idx in treatment.index: 
                controls_within_caliper = control[abs(PS.loc[t_idx] - PS.loc[control.index]) <= caliper] # Get the control units within the propensity score caliper
                controls_within_caliper_hc = [c_idx for c_idx in controls_within_caliper.index if all(X.loc[t_idx, f] == X.loc[c_idx, f] for f in hard_constraints)] # filter the controls by those where the hard constraint features have the same value for treated and control in X
                prob += pulp.lpSum(x[(t_idx, c_idx)] for c_idx in controls_within_caliper_hc) >= num_matches

            # Constraint 2: each control unit is matched to at most one treatment unit
            for c_idx in control.index:
                prob += pulp.lpSum(x[(t_idx, c_idx)] for t_idx in treatment.index) <= 1

            # Solve the optimization problem
            start_time = time.time()  # Record the start time
            prob.solve(pulp.GLPK_CMD(msg=0)) # Use GLPK solver and suppress output
            end_time = time.time()  # Record the end time
            execution_time = end_time - start_time # Calculate the execution time
            print("Execution Time:", execution_time, "seconds")

            # Retrieve the matches from the decision variables 
            match_num = 1  # Initialize the match number
            matched_treatment = set()  # Keep track of treatment units that have been matched
            for t_idx in treatment.index: # Iterate over the treatment units
                matched_control = []  # Keep track of control units matched with this specific treatment unit
                for c_idx in control.index: # Iterate over the control units
                    if pulp.value(x[(t_idx, c_idx)]) == 1: # If the decision variable is 1, the pair is a match
                        matched_control.append(c_idx) # Add the control unit index to the matched control list
                if matched_control: # If the matched control list is not empty
                    if t_idx not in matched_treatment: # If the treatment unit has not been matched yet, it is not in the set
                        for c_idx in matched_control: # Iterate over the matched control units to this specific treatment unit
                            matches.append([t_idx, c_idx, match_num]) # Add the matched pair to the matches list with the match number
                        matched_treatment.add(t_idx) # Add the treatment unit index to the matched treatment set to keep track of matched treatment units
                        match_num += 1 # Increment the match number
    
    # If the problem found matches then return the matches
    if matches:
        match_full = pd.DataFrame(columns=X.columns)  # Create an empty dataframe to store the matches including the features from X

        # Iterate over the matches and include the matched number column 
        for match in matches:
            match_treatment = treatment.loc[match[0]].to_frame().T # Get matched treatment unit features values from X
            match_control = control.loc[match[1]].to_frame().T # Get matched control unit features values from X
            match_treatment['match_num'] = match[2]  # Use the match number from the matches list
            match_control['match_num'] = match[2]  # Use the match number from the matches list
            match_full = pd.concat([match_full, match_treatment, match_control]) # Concatenate the matched treatment and control units to the match_full dataframe
                
        # Add a column for the treatment indicator from T
        match_full['treatment'] = T[match_full.index]

        #drop duplicated treatment rows that are added for when num_matches>1 (just for formatting purposes)
        match_full=match_full[~((match_full['treatment']==1) & (match_full['match_num'].duplicated()))]
        print('Matching Process Complete!')
        return match_full
    
    # If the problem did not find matches, then print a suggestion to relax the constraints
    elif not matches:
        print('** WARNING ** No matches found. You should try to relax your constraints in the optimization problem as the problem may be infeasible. Also, you can try to reduce the number of control matches as there might not be enough control units to match to.')
        return None


#########################################################################################
# 6. Evaluation
#########################################################################################
# Functions: asmd

def asmd(data, features, before, T='treatment', perspective=['individual', 'global']):
    '''
    Calculates the average standardized mean difference for each feature. ASMD = Abs(Mean Treated - Mean Control)
    
    Parameters:
    -----------
    data: pd.DataFrame
        The feature matrix with the treatment. 
    features: list
        The list of important features to be used for the balance calculation.
    before: boolean
        Whether you are calculating the ASMD before or after matching.
    T: str, optional
        The name of the treatment column (default: 'treatment').
    perspective: str
        The perspective from which to calculate the ASMD. Options are 'individual' or 'global'. Individual calculates the ASMD for each match, and global calculates the ASMD for the entire dataset.
    
    Returns:
    --------
    ASMD_by_feature: pd.Series
        A series with the average standardized mean difference for each feature. 
    '''
    # Identify treatment and control units    
    treatment = data[data[T] == 1] 
    control = data[data[T] == 0] 

    if perspective=='global': # Calculate ASMD globally comparing treatment and control
        ASMD_by_feature = abs(treatment[features].mean() - control[features].mean()) # calculate ASMD for each feature
        ASMD_global=ASMD_by_feature.mean() # calculate global ASMD as the mean of the ASMDs for each feature
        if before:
            print(f'ASMD before matching: {ASMD_global}')
        else:
            print(f'ASMD after matching: {ASMD_global}')
        return ASMD_by_feature
    
#########################################################################################
# 7. Difference in Difference
#########################################################################################
# Functions: diff_in_diff

def diff_in_diff(df, outcome, treatment):
    '''
    Performs the difference in difference analysis on the matched data.
        
    Parameters:
    -----------
    df : dataframe
        Dataframe with matched units.
    outcome : string
        Name of outcome variable.
    treatment : string
        Name of treatment variable.
        
    Returns:
    --------
    None - prints the results of the difference in difference analysis and plots the results.
    '''
    # Filter the dataframe into the treatment group and control group
    treatment_group = df[df[treatment] == 1]
    control_group = df[df[treatment] == 0]

    # Calculate the difference in difference 
    diff_in_diff_num = (treatment_group[treatment_group['Post'] == 1][outcome].mean() - treatment_group[treatment_group['Post'] == 0][outcome].mean()) - (control_group[control_group['Post'] == 1][outcome].mean() - control_group[control_group['Post'] == 0][outcome].mean())
                            
    # Fit the model
    # Post = 0 means 2020, Post = 1 means 2021
    # SURVEY_CRIMSI = 0 means control, SURVEY_CRIMSI = 1 means treatment
    # Year*SURVEY_CRIMSI is the interaction term, it's the difference in difference as it's only 1 when Year = 1 and SURVEY_CRIMSI = 1
    model = ols(outcome+' ~ '+treatment+' + Post + Post*'+treatment, data=df1).fit()
    # Print the results
    print(diff_in_diff_num, "with significance level of", model.pvalues['Post:'+treatment])

    print(model.summary())
    
    # Extract the relevant columns for plotting
    years = ['Pre', 'Treatment Applied', 'Post']
    # get the mean of outcome when post = 1 and post = 0 for both treatment and control
    control_meets = [control_group[control_group['Post']==0][outcome].mean(), control_group[control_group['Post']==1][outcome].mean()]
    treatment_meets = [treatment_group[treatment_group['Post']==0][outcome].mean(), treatment_group[treatment_group['Post']==1][outcome].mean()]
    treatment_meets = np.insert(treatment_meets, 1, (control_meets[1] - control_meets[0])/2 + treatment_meets[0])
    control_meets = np.insert(control_meets, 1, (control_meets[1]+control_meets[0])/2)

    plt.figure(2)
    # Plotting
    plt.plot(years, control_meets, label='Control Group', color='orange')
    plt.plot(years, treatment_meets, label='Treatment Group', color='royalblue')

    # create a line at with the same slope as control_meets but with the y-intercept of treatment_meets
    plt.plot(years, [control_meets[0] + (treatment_meets[0] - control_meets[0]),(treatment_meets[0]+control_meets[2] + (treatment_meets[0] - control_meets[0]))/2 , control_meets[2] + (treatment_meets[0] - control_meets[0])], linestyle='--', label='Projected', color='royalblue')

    # Add labels and title
    plt.ylabel(outcome)
    plt.title('Difference in Difference Analysis')

    # Add legend
    plt.legend()

    # Display the plot
    plt.show()

#####################################################################################################################
#                                             QUESTIONS ON SCRIPT
#####################################################################################################################
# Functions: PS_inquiry, dist_matching_procedure, matching_questions

def PS_inquiry(data, T, y, outcome):
    '''
    This function walks through the propensity score process.
    
    Parameters:
    -----------
    data : dataframe
        Dataframe with treatment and control units.
    T : pd.series
        Series with treatment indicator. 1 if treatment, 0 if control.
    y : pd.series
        Series with outcome variable.
    outcome : string
        Name of outcome variable.
    
    Returns:
    --------
    matches : dataframe
        Dataframe with matched units.
    '''
    # factor analysis using the 1:10 rule of thumb
    fa = factor_analysis(data, T, k=min(len(data[T==1]), len(data[T==0]))//10)[0]
    fa[outcome]=y
    fa['treatment']=T
    # get propensity scores
    df_ps=ps_balanced(fa.iloc[:,:-1], fa.iloc[:,-1])
    data['propensity_score']=df_ps['FINAL_PS']
    
    # ask if there are any hard constraints
    questions = [
    inquirer.List('d',
                  message="Are there any features that must be an exact match?",
                  choices=['Yes', 'No']),
    ]
    
    answers = inquirer.prompt(questions)
    
    hc = None
    
    choice = data.columns.tolist()[:-2]
    
    def sort_choice(item):
        if 'GRADE' in item or 'Subject' in item:
            return 0  # Move strings containing 'GRADE' or 'Subject' to the front
        else:
            return 1  # Keep other strings in their original order

    sorted_choice = sorted(choice, key=sort_choice)
    
    
    # if there are hard constraints, ask which ones
    if answers['d'] == "Yes":
        print('Select any features that must be an exact match?')
        questions2 = [
            inquirer.Checkbox('hard_constraints',
                        message='Hit space to select features and enter to submit',
                        choices=sorted_choice,
                        )
            ]
        hard_constraints = inquirer.prompt(questions2)
        hc = hard_constraints['hard_constraints'] 
        
    # ask matching questions
    matching, replacement, optimal = matching_questions()
        
    # create a distance matrix using the difference in propensity scores
    dist = ps_distance_matrix(data,T,df_ps['FINAL_PS'], hc)
    # run matching
    matches = dist_matching(data,T, dist, num_matches = matching, with_replacement = replacement, optimal = optimal)
    # return matches and columns used for use during the ASMD calculation
    return matches, X_trans.columns.tolist()[:-2]


def dist_matching_procedure(data,T, y, C):
    '''
    This function walks through the distance matching procedure.
    
    Parameters:
    -----------
    data : dataframe
        Dataframe with treatment and control units.
    T : pd.series
        Series with treatment indicator. 1 if treatment, 0 if control.
    y : pd.series
        Series with outcome variable.
    C : boolean
    
    Returns:
    --------
    matches : dataframe
        Dataframe with matched units.
    total_features : list
        List of features used in matching.
    '''
    # ask if user wants to match on all features or a subset
    #print("Would you like to match on all features")
    questions = [
        inquirer.List(
            'a',
            message="What features would you like to match on?",
            choices=['All features', 'Subset of important features']
        )
    ]
    answers = inquirer.prompt(questions)
    user_choice = answers['a']
    weights = None

    total_features = None
    if user_choice == "Subset of important features":
        # use feature selection to select the initial feature subset
        # if columns contains NaN, print the columns
        total_features, coef_all,coef_subset = feature_selection(data, y)
        # print features and coefficients
        print("Here are the selected features:")
        for f in total_features:
            print('\t'+f)
        print('\n')
        
        # ask if there are any other features that should be included
        print('Are there any other features you think are important to predicting '+ y.name +' that are not included?')
        questions2 = [
        inquirer.Checkbox('selected_features',
                      message= 'Hit space to select features and enter to submit',
                      choices=list(set(data.columns.tolist()) - set(total_features))
                      )
        ]
        features_to_include = inquirer.prompt(questions2)
        # if user selects features to include, re-run feature selection with those features included
        if features_to_include['selected_features']:
            total_features  = feature_selection(data, y, must_include = features_to_include['selected_features'])[0]
        # print final selected features
        print("Final selected features:")
        for f in total_features:
            print('\t'+f)
        print('\n')
        
        print('If you think any of the selected features are more important than others, you can weight them by high, medium, or low.')
        # ask if user wants to weight features
        questions = [
        inquirer.List(
            'a',
            message='Would you like to weight the features or let them all be equal?',
            choices=['Weighted by importance', 'Equal weights']
        )]
        weighted = inquirer.prompt(questions)
        
        # if user wants to weight features, ask which ones they want to weight as high, medium, and low
        if weighted['a'] == 'Weighted by importance':
            print('Select the features you would like to weight as \'HIGH\':')
            questions2 = [
                inquirer.Checkbox('weights',
                            message='Hit space to select features and enter to submit',
                            choices=total_features,
                            )
                ]
            weights = inquirer.prompt(questions2)
            high = weights['weights']
            print('Select the features you would like to weight as \'MEDIUM\':')
            questions2 = [
                inquirer.Checkbox('weights',
                            message='Hit space to select features and enter to submit',
                            choices=list(filter(lambda x: x not in high, total_features)),
                            )
                ]
            print('The remaining features will be weighted as \'LOW\'')
            weights = inquirer.prompt(questions2)
            medium = weights['weights']
                        
            # create weights list where high features are weighted 1.5, medium 1, and low .5
            weights = [1.5 if f in high else 1 if f in medium else .5 for f in total_features] 
        else:
            weights = None
    # else we use all features, meaning we don't weight any features
    else:
        total_features = data.columns.tolist()
        weights = None
    
    # ask if there are any hard constraints
    questions = [
    inquirer.List('d',
                  message="Are there any features that must be an exact match?",
                  choices=['Yes', 'No']),
    ]
    
    answers = inquirer.prompt(questions)
    
    choice = data.columns.tolist()[:-2]
    
    def sort_choice(item):
        if 'GRADE' in item or 'Subject' in item:
            return 0  # Move strings containing 'GRADE' or 'Subject' to the front
        else:
            return 1  # Keep other strings in their original order

    sorted_choice = sorted(choice, key=sort_choice)
    
    hard_constraints = None
    
    # if there are hard constraints, ask which ones
    if answers['d'] == "Yes":
        print('Select any features that must be an exact match?')
        questions2 = [
            inquirer.Checkbox('hard_constraints',
                        message='Hit space to select features and enter to submit',
                        choices=sorted_choice,
                        )
            ]
        hard_constraints = inquirer.prompt(questions2)
        hard_constraints = hard_constraints['hard_constraints'] 

    # create distance matrix
    dist = distance_matrix_hard_constraints(data, T, features=total_features,hard_constraint_features=hard_constraints, weights=weights)

    # ask matching questions
    matching, replacement, optimal = matching_questions()

    # if matchhing with calipers
    if C:
        # factor analysis using the 1:10 rule of thumb
        fa = factor_analysis(data, T, k=min(len(data[T==1]), len(data[T==0]))//10)[0]
        fa[outcome]=y
        fa['treatment']=T
        # get propensity scores
        df_ps=ps_balanced(fa.iloc[:,:-1], fa.iloc[:,-1])
        # run matching with calipers
        matches = dist_matching_PS_calipers(data, T, df_ps['FINAL_PS'], dist, with_replacement=replacement, optimal=optimal, num_matches=matching, caliper=.02, hard_constraints=hard_constraints)
        # add propensity scores to matches based on index
        matches['propensity_score'] = df_ps['FINAL_PS']
    else:
        # run matching
        matches = dist_matching(data,T, dist, num_matches = matching, with_replacement = replacement, optimal = optimal)
    return matches, total_features

def matching_questions():
    '''
    This function asks the user how they would like to match.
    
    Parameters:
    -----------
    None
    
    Returns:
    --------
    matching : int
        Number of matches per treatment unit.
    replacement : boolean
        Whether or not to match with replacement.
    optimal : boolean
        Whether or not to use optimal matching.
    '''
    # ask number of matches per treatment unit and whether or not to match with replacement
    questions = [inquirer.List('One-to-one or one-to-many',
                    message="How many control matches per treatment row?",
                    choices=[1,2,3,4,5]),
                    inquirer.List('with or without replacement',
                        message="Do you want to match with or without replacement?",
                        choices=['with', 'without'])         
        ]
    answers = inquirer.prompt(questions)
    matching = answers['One-to-one or one-to-many']
    replacement = answers['with or without replacement']
    # if user selects without replacement, ask if they want to match optimally or use KNN
    if replacement == "without":
        replacement = False
        questions2 = [inquirer.List('optimal or not',
                    message="Would you like to match optimally(takes longer) or use KNN?",
                    choices=['optimal', 'not optimal'])
        ]
        answers = inquirer.prompt(questions2)
        optimal = answers['optimal or not']
        if optimal == "optimal":
            optimal = True
        else:
            optimal = False
    else:
        replacement = True
        optimal = False
    # return these preferences
    return matching, replacement, optimal 


#####################################################################################################################
#                                                 MAIN
#####################################################################################################################
if __name__ == '__main__':
    # Read in the command line arguments and throw an error if the number of arguments is incorrect
    if len(sys.argv) != 4:
        print("Usage: python script.py <csv_file> <treatment_column_name> <outcome_column_name>")
        sys.exit(1)

    # Read in the data
    csv_file = sys.argv[1]
    treatment = sys.argv[2]
    outcome = sys.argv[3]
    
    # Check if the file exists
    if not os.path.isfile(csv_file):
        print("File does not exist")
        sys.exit()

    # Read in the data
    X, T, y = read_data(path=csv_file, outcome=outcome, treatment=treatment)
    #X = X.drop(['STAAR_M_%OnlyApproaches', 'STAAR_M_%OnlyMasters', 'STAAR_M_%OnlyMeets', 'STAAR_M_APPROACHES', 'STAAR_M_MASTERS', 'STAAR_M_MEETS', 'STAAR_M_STAAR_PROG_MEAS', 'STAAR_R_%OnlyApproaches', 'STAAR_R_%OnlyMasters', 'STAAR_R_%OnlyMeets', 'STAAR_R_APPROACHES', 'STAAR_R_MASTERS', 'STAAR_R_MEETS', 'STAAR_R_STAAR_PROG_MEAS'],axis=1)
    # Ask user if they want to drop any columns
    print("Are there any columns that should be removed from the matching? We suggest removing any variables related to " + outcome + " and any ID variables.")
    print("Type yes or no. You can also copy in a list of columns here if you have.")
    val = input("Type here: ")
    
    # If the user wants to drop columns, ask them which ones
    if val.lower().lstrip() == 'yes':
        questions2 = [
                inquirer.Checkbox('to_drop',
                            message='Hit space to select features to remove and enter to submit',
                            choices=X.columns.tolist(),
                            )
                ]
        drop = inquirer.prompt(questions2)
        # Output the list of columns to drop so the user can copy it for next time
        print("Here is a list of what you dropped, you can copy this for next time")
        print(drop['to_drop'])
        drop = drop['to_drop']
    # if the user doesn't want to drop any columns, set drop to None
    elif val.lower().lstrip() == 'no':
        drop = None
    # elif val.lower() contains [' - i.e. is a list of columns to drop
    elif (val.lower().lstrip()).startswith('['):
        drop = ast.literal_eval(val.lstrip())
    # if the user wants to drop a list of columns, convert the string to a list
    else:
        print("I couldn't quite understand that. Please try again. Your options are yes, no, or a list of columns to drop.")
        sys.exit()
    
    # Copy the data so we can transform it without affecting the original data
    X_2 = X.copy()
    # Drop the columns the user specified
    if drop != None:
        X_2 = X.drop(drop,axis=1)
        #['STAAR_masked_DISTRICT', 'STAAR_masked_CAMPUS', 'ROSTER_masked_STAFF_ID', 'ROSTER_SERVICE_ID', 'ROSTER_Translation', 'ROSTER_SCHOOL_YEAR', 'ROSTER_STAFF_TYPE_CODE', 'STAAR_Approaches', 'STAAR_Masters','STAAR_Meets_2021']
    # Transform the data using the function we wrote
    X_trans = transform_data(X_2)
    print("\n")
    
    # Ask the user which matching procedure they want to use
    questions = [inquirer.List('match',
                    message="How would you like to match?",
                    choices=['Propensity Scores', 'Distance','Distance with PS Calipers'])]
    answers = inquirer.prompt(questions)
    match = answers['match']
    
    # Run the matching procedure the user specified
    if match == "Propensity Scores":
        matches,total_features = PS_inquiry(X_trans, T, y, outcome)
    elif match == "Distance":
        matches,total_features = dist_matching_procedure(X_trans,T,y,False)
    else:
        matches,total_features = dist_matching_procedure(X_trans,T,y,True)
    
    # write transformed matches to csv
    matches.to_csv('matches_transformed.csv', index=True)
    
    X[outcome] = y
    # get the original data for the matched units and write to csv
    ogmatches = get_original_data(matches, X)
    ogmatches.to_csv('matches_original.csv', index=True)
    print("Matches written to matches_transformed.csv and matches_original.csv")
    print("\n")
    
    X['treatment'] = T
    X_trans['treatment'] = T
    
    # Calculate the ASMDs of the original data and the matched data
    y = asmd(X_trans,total_features, perspective='global', before=True)
    x = asmd(matches,total_features, perspective='global', before=False)

    # plot the ASMDs - line graph for each feature
    plt.figure(1)
    plt.plot(y, label='ASMD before matching', color='green')
    plt.plot(x, label='ASMD after matching', color='purple')
    plt.title('ASMD before and after matching')
    plt.ylabel('ASMD')
    plt.xlabel('Features')
    plt.legend()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.savefig('ASMD.png')
    print("*ASMDs before and after saved to ASMD.png")
    print("\n")
    print("Would you like to run a difference in difference analysis and evaluate the matches?")
    questions = [inquirer.List('diff',
                    message="Select yes or no",
                    choices=['Yes', 'No'])]
    answers = inquirer.prompt(questions)
    diff = answers['diff']
    print("\n")
    if diff == "No":
        sys.exit()
    
    # Read in the matches and do some transformations so that we can use them for the difference in difference analysis
    X, T, y = read_data(path='matches_original.csv', outcome=outcome, treatment='treatment')
    
    choice = X.columns
    # remove 'Unnamed: 0' from choice
    choice = list(filter(lambda x: x != 'Unnamed: 0', choice))
    
    def sort_choice(item):
        if outcome in item:
            return 0  # Move strings containing 'GRADE' or 'Subject' to the front
        else:
            return 1  # Keep other strings in their original order

    sorted_choice = sorted(choice, key=sort_choice)
    
    print('Select your pre-treatment outcome variable')
    questions2 = [
        inquirer.Checkbox('pre',
                    message='Hit space to select feature and enter to submit',
                    choices=sorted_choice,
                    )
        ]
    
    pre = inquirer.prompt(questions2)
    pre = pre['pre']
    pre = pre[0]

    df = pd.DataFrame()
    df[outcome] = y
    df[pre] = X[pre]
    df[treatment] = T
        
    # Create a new dataframe and adjust the values for the difference in difference analysis
    df1 = df.copy()
    df1['Post'] = 1
    # remove STAAR_Meets_2021 and append those values to STAAR_Meets
    df2 = pd.DataFrame()
    df2[outcome] = df[pre]
    df2[treatment] = df[treatment]
    df2['Post'] = 0
    # Check the column names in df1 and df2
   
    df1 = df1.drop([pre], axis=1)
    df1 = pd.concat([df1, df2])
    
    # Run the difference in difference analysis for both subjects
    print("Difference in Difference Analysis")
    diff_in_diff(df1,outcome, treatment)
    # Show all the plots in the same window