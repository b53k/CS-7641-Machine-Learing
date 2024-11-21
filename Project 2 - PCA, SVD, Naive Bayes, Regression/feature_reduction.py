import pandas as pd
import statsmodels.api as sm
from typing import List


class FeatureReduction(object):
    def __init__(self):
        pass

    @staticmethod
    def forward_selection(
        data: pd.DataFrame, target: pd.Series, significance_level: float = 0.05
    ) -> List[str]:  # 9 pts
        """
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            forward_list: (python list) contains significant features. Each feature
            name is a string
        """
        #raise NotImplementedError
        forward_list = []
        init_features = data.columns.tolist()

        while len(init_features) > 0:
            initial_set = set(init_features)
            forward_set = set(forward_list)

            remaining_features = list(initial_set - forward_set)
            pvalue_new = pd.Series(index=remaining_features)

            for i in remaining_features:
                # add bias terms as suggested in TIP 3
                add_bias = sm.add_constant(data[forward_list+[i]])
                # fit the regression model (TIP 3)
                least_squares = sm.OLS(target, add_bias).fit()
                pvalue_new[i] = least_squares.pvalues[i]
            # find minimum p-value
            min_pvalue = pvalue_new.min()
            # select feature with minimum p-value
            if(min_pvalue < significance_level):
                forward_list.append(pvalue_new.idxmin())
            else:
                break

        return forward_list


    @staticmethod
    def backward_elimination(
        data: pd.DataFrame, target: pd.Series, significance_level: float = 0.05
    ) -> List[str]:  # 9 pts
        """
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            backward_list: (python list) contains significant features. Each feature
            name is a string
        """
        #raise NotImplementedError

        # convert feature matrix to list
        backward_list = data.columns.to_list()

        while len(backward_list) > 0:
            # add a columns of ones to an array ---> prepend
            bias_features = sm.add_constant(data[backward_list])
            # perform ordinary least squares but neglect for the bias to get p-values
            p_values = sm.OLS(target, bias_features).fit().pvalues[1:] # Idea suggested on Ed Discussion post
            # get maximum p-value
            max_p_value = p_values.max()
            # select feature with maximum p-value to be removed
            if significance_level > max_p_value:
                break
            else:
                delete_feature = p_values.idxmax()
                # remove feature from the backward_list
                backward_list.remove(delete_feature)

        return backward_list