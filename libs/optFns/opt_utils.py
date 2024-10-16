import pandas as pd
import numpy as np
import scipy
import scipy.optimize as sco
import datetime
from libs.signal_utils import sharpe_ratio, check_and_clean_data
from pandas.tseries.offsets import DateOffset
import concurrent.futures
from cvxopt import matrix, solvers


def generate_date_ranges(timeseries, frequency = "years", RollRec = "Rec", min_window=3, RollYear=10, FullSample = False):
    dates = []
    if FullSample:
        return([timeseries.index[0],timeseries.index[-1]])

    if RollRec == "Rec":
        start_date = timeseries.index[0]
        end_date = start_date + DateOffset(years=min_window)
        while end_date <= timeseries.index[-1]:
            if(end_date not in timeseries.index):
                matching_dates = timeseries.index[timeseries.index.year == end_date.year]
                end_date = matching_dates[matching_dates.month == end_date.month][0]
            dates.append((start_date, end_date))
            if frequency == "years":
                end_date += DateOffset(years=1)
            elif frequency == "months":
                end_date += DateOffset(months=1)

    elif RollRec == "Rolling":
        start_date = timeseries.index[0]
        while start_date + DateOffset(years=RollYear) <= timeseries.index[-1]:
            end_date = start_date + DateOffset(years=RollYear)
            dates.append((start_date, end_date))
            if frequency == "years":
                start_date += DateOffset(years=1)
            elif frequency == "months":
                start_date += DateOffset(months=1)

    return dates

def RollingOpt(pnls, dates, prior_strength, prior = None, sr_cut= 0, fwd_looking = False, threshold = 0.3, doRound = False,scaleTo = None):
    """

    :param pnls: assumes an arithmetic series, indexed at 100.
    :param dates:
    :param prior:
    :param prior_strength:
    :param sr_cut:
    :param fwd_looking:
    :param threshold:
    :param doRound:
    :param scaleTo:
    :return:
    """
    #prior_strength = 0.5
    end_dates = [dd[1] for dd in dates]
    priorNames = pnls.columns
    assetPriors = np.repeat(1/len(priorNames), len(priorNames))
    x = assetPriors/sum(assetPriors)
    # If you
    if prior:
        prior = prior
    else:
        prior = pd.DataFrame(data = [x] , index = [dates[0][1]], columns = pnls.columns )
    prior_strength_df = pd.DataFrame(data = [np.repeat(1, prior.shape[0])], index = [dates[0][1]])*prior_strength
    rets = pnls.diff(1).fillna(0)
    # add priors
    filled_prior = prior.reindex(end_dates)
    filled_prior.ffill(inplace=True)
    prior_strength_df = prior_strength_df.reindex(end_dates)
    prior_strength_df.ffill(inplace=True)
    wgts_list = []
    for dd in dates:
        start_date, end_date = dd
        wgts = getFolioWgts(start_date, end_date, rets, prior, prior_strength, threshold=0.3, doRound=False,
                     scaleTo=None)
        wgts_list.append(wgts)

    historical_wgts = pd.concat(wgts_list)
    return historical_wgts

def holdAll( dates, rets, prior, prior_strength):
    # do some multithreading?
    def task_wrapper(date_tuple):
        """Wrapper for getFolioWgts to handle additional fixed parameters."""
        start_date, end_date = date_tuple
        return getFolioWgts(start_date, end_date, rets, prior, prior_strength, threshold=0.3, doRound=False,
                            scaleTo=None)

    # Using ThreadPoolExecutor to run getFolioWgts in parallel for each date tuple
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Schedule the execution of getFolioWgts for each date tuple and handle additional parameters through a wrapper
        futures = [executor.submit(task_wrapper, date_tuple) for date_tuple in dates]

        # Iterating over the results as they are completed
        for future in concurrent.futures.as_completed(futures):
            try:
                # Retrieve result
                result = future.result()
                # Handle result (e.g., print or store it)
                print(result)
            except Exception as e:
                # Handle potential exceptions
                print(f"An error occurred: {e}")
        # now you produce t

# Find efficient frontier
def findFronty(postExpRet, postCovar):
    # Assuming 'mean_returns' is your N by 1 matrix (vector) of mean returns
    # and 'cov_matrix' is your N by N covariance matrix of returns
    mean_returns = matrix(postExpRet.values)  # N by 1 matrix of mean returns
    cov_matrix = matrix(postCovar.values)  # N by N covariance matrix

    # Number of assets
    n = mean_returns.size[0]

    # Convert mean returns and covariance matrix to cvxopt matrices
    P = cov_matrix
    q = matrix(0.0, (n, 1))  # Minimize 1/2 x^T P x without a linear term
    ones_vector = np.ones((1, n))
    # Equality constraint Ax = b
    # A is a stacked matrix of the mean returns' transpose and a row vector of ones for the sum of weights
    A = matrix(np.vstack((mean_returns.T, ones_vector)))
    b = matrix([max(mean_returns), 1.0])  # target_return is the desired portfolio return, 1.0 for the sum of weights

    # Inequality constraint Gx <= h (no short selling, weights between 0 and 1)
    G = matrix(-np.eye(n))  # Negative identity matrix for non-negativity constraint
    h = matrix(0.0, (n, 1))  # Vector of zeros
    solvers.options['show_progress'] = False
    # Solve the quadratic programming problem
    sol = solvers.qp(P, q, G, h, A, b)

    # Extract the optimal portfolio weights
    optimal_weights = np.array(sol['x'])

    return optimal_weights


def getFrosty(nCols, nRows,SampleReturns,SampleCov, PriorWeights,taco,nobo):
    """
    :param nCols:
    :param nRows:
    :param SampleReturns:
    :param SampleCov:
    :param PriorWeights:
    :param taco:
    :param nobo:
    :return:
    """
    # N by 1
    ones = np.ones((1, nCols)).T
    pWeights = np.array(PriorWeights).T
    sumPriorWeights = np.sum(pWeights)
    megaTaco = taco/(taco+nRows)
    megaNobo = nobo/(nobo+nRows)
    SampleReturns = pd.DataFrame(SampleReturns)
    RSamExpRet = SampleReturns*((pWeights*(nCols/sumPriorWeights)))
    # Convert PriorWeights to a diagonal matrix
    diagPriorWeights = np.diag(PriorWeights.iloc[0].values)
    # Perform the matrix operations
    scaling_factor = (nCols / sumPriorWeights) ** 2
    RSamCov = (diagPriorWeights@SampleCov@diagPriorWeights)* scaling_factor
    # Create a vector of ones
    #ones = np.ones((nCols, 1))

    # Perform the operations: Calculate the weighted average of RSamExpRet and divide by N
    priorMega = np.dot(ones.T, RSamExpRet) / nCols
    # Since priorMega is a 1x1 matrix, convert it to a scalar (numeric)
    priorMega = priorMega.item()
    RSM = RSamCov + (RSamExpRet.values-priorMega*ones) @ (RSamExpRet.values-priorMega*ones).T
    priorS2 = sum(np.diag(RSM))/nCols
    total_sum_RSM = np.dot(np.dot(ones.T, RSM), ones)
    # Calculate sum(diag(RWSM)), which is the sum of the diagonal elements of RWSM
    sum_diag_RWSM = np.sum(np.diag(RSM))
    priorSJ = (total_sum_RSM - sum_diag_RWSM) / (nCols * (nCols - 1)) #(t(ones)%*%RSM%*%ones-sum(diag(RWSM)))/(n(N-1))
    priorR = priorSJ/priorS2
    # construct emp priors
    priorSig =  (np.sqrt(priorS2)/nCols)*np.diag(1/PriorWeights.iloc[0].values)
    priorExpRet = (priorMega/nCols)*(1/pWeights) # Pw must be an N x 1 matrix, where N is nCols of return stream
    priorCorrel = priorR*ones @ ones.T + (1-priorR)*np.eye(nCols)
    priorCovar = priorSig@priorCorrel@priorSig
    postExpRet = megaTaco*priorExpRet+ (1-megaTaco)*SampleReturns
    postCovar = ((nobo+nRows)/(nobo+nRows-2))*(1+1/(taco+nRows))*(megaNobo*priorCovar + (1-megaNobo)*SampleCov + megaTaco*(nRows/(nobo+nRows))*((SampleReturns-priorExpRet) @ (SampleReturns-priorExpRet).T))
    return postExpRet, postCovar


def getFolioWgts(start_date, end_date, rets,prior_filled,prior_strength, sr_cut = 0, threshold = 0.3, doRound = False,scaleTo = None):
    """

    :param startt_date:
    :param end_date:
    :param rets:
    :param prior:
    :param prior_strength:
    :param threshold:
    :param doRound:
    :param scaleTo:
    :return:
    """
    # Check sr cut
    # TODO: need to handle nothing being selected, perhaps a continution? next statement?
    fullRets= rets.loc[start_date:end_date]
    sampleRets = fullRets
    if sr_cut >= 0:
        # We pass in undifferenced pnls
        sr_val = sharpe_ratio(sampleRets)
        selected_pnls = sr_val[sr_val >= sr_cut]
        selection = sampleRets[selected_pnls.index]
        unselected_cols = sampleRets.columns.difference(selection.columns)
        unsel = sampleRets[unselected_cols]

    # redo priors
    selPriorNames = selection.columns.tolist()
    selAssetPriors = np.repeat(1/len(selPriorNames), len(selPriorNames))
    selx = selAssetPriors/sum(selAssetPriors)
    selPrior = pd.DataFrame(data = [selx] , index = [end_date], columns = selection.columns )
    selPrior_strength_df = pd.DataFrame(data = [np.repeat(1, selPrior.shape[0])], index = [end_date])*prior_strength
    # get covariance martix
    covMatrix = selection.cov()
    retsMean = selection.mean()
    mega = selPrior_strength_df.values.flatten()
    prior_len = selection.shape[0]*mega/(1-mega)
    # get post returns and covars
    posMean , posCo = getFrosty(nCols = selection.shape[1], nRows = selection.shape[0], SampleReturns = retsMean,SampleCov = covMatrix, PriorWeights = selPrior,taco = selection.shape[0],nobo = selection.shape[0])
    wgts = findFronty(posMean, posCo)
    wgts_df = pd.DataFrame(wgts.T, columns=selection.columns, index=[end_date])
    if unsel.shape[1] > 0:
        unsel_strats = pd.DataFrame((unsel.loc[end_date])*0, columns=unsel.columns, index=[end_date])
        final_wgts = pd.concat([wgts_df.round(6),unsel_strats.fillna(0)], axis = 1)
    else:
        final_wgts = wgts_df.round(6)
    return final_wgts.reindex(sorted(final_wgts.columns), axis=1)



# Now target_returns and target_volatilities contain the points on the efficient frontier
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    outDir = r"C:\Users\edgil\Documents\git_working\Research\pcode\tests\dummy_data\\"
    pnls = pd.read_csv(outDir + "pnls.csv", index_col=0, parse_dates=True)
    dts = generate_date_ranges(pnls, min_window = 5, FullSample=False)
    weights = RollingOpt(pnls, dates = dts, prior_strength= 0.75)
     # TODO: Add in a util type regression model please. scaled retursn and a vector of 1's , betas are the weights, its a SR max approach.