import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
import argparse

'''Demo for creating customized multi-class focal loss objective function.  
https://github.com/dmlc/xgboost/blob/1d22a9be1cdeb53dfa9322c92541bc50e82f3c43/src/objective/multiclass_obj.cu
'''


np.random.seed(2020)

kRows = 100
kCols = 10
kClasses = 4                    # number of classes

kRounds = 10                    # number of boosting rounds.

# Generate some random data for demo.
X = np.random.randn(kRows, kCols)
y = np.random.randint(0, 4, size=kRows)

m = xgb.DMatrix(X, y)


def softmax(x):
    '''Softmax function with x as input vector.'''
    e = np.exp(x)
    return e / np.sum(e)

def focal_logloss_derivative_gamma2(asample_class_prob,target_label):
    gamma=2
    target = target_label
    # print('pt=p[target]:',target)
    p = asample_class_prob
    pt=p[target]

    kClasses=len(asample_class_prob)
    assert target >= 0 or target <= kClasses

    grad = np.zeros(kClasses, dtype=float)
    hess = np.zeros(kClasses, dtype=float)
    eps = 1e-6

    for c in range(kClasses):
        pc=p[c]
        if c == target:
            g=(gamma * np.power(1-pt,gamma-1) * pt * np.log(pt) - np.power(1-pt,gamma) ) * (1 - pc)
            h = (-4*(1-pt)*pt*np.log(pt)+np.power(1-pt,2)*(2*np.log(pt)+5))*pt*(1-pt)
        else:
            g=(gamma * np.power(1-pt,gamma-1) * pt * np.log(pt) - np.power(1-pt,gamma) ) * (0 - pc)
            h = pt*np.power(pc,2)*(-2*pt*np.log(pt)+2*(1-pt)*np.log(pt) + 4*(1-pt)) - pc*(1-pc)*(1-pt)*(2*pt*np.log(pt) - (1-pt))
        grad[c] = g
        hess[c] = max(h,eps)
    return grad,hess


def log_focal_loss_obj(preds: np.ndarray, dtrain: xgb.DMatrix):
    labels = dtrain.get_label()
    # print(preds.shape)
    kRows, kClasses = preds.shape

    if dtrain.get_weight().size == 0:
        # Use 1 as weight if we don't have custom weight.
        weights = np.ones((kRows, 1), dtype=float)
    else:
        weights = dtrain.get_weight()

    grad = np.zeros((kRows, kClasses), dtype=float)
    hess = np.zeros((kRows, kClasses), dtype=float)


    for r in range(kRows):
        #print(preds[r])
        target = int(labels[r])
        assert target >= 0 or target <= kClasses
        p = softmax(preds[r, :])
        grad_r,hess_r = focal_logloss_derivative_gamma2(p,target)
        grad[r]=grad_r*weights[r]
        hess[r]=hess_r*weights[r]

    # Right now (XGBoost 1.0.0), reshaping is necessary
    grad = grad.reshape((kRows * kClasses, 1))
    hess = hess.reshape((kRows * kClasses, 1))

    return grad, hess


def predict(booster, X):
    '''A customized prediction function that converts raw prediction to
    target class.
    '''
    # Output margin means we want to obtain the raw prediction obtained from
    # tree leaf weight.
    predt = booster.predict(X, output_margin=True)
    out = np.zeros(kRows)
    for r in range(predt.shape[0]):
        # the class with maximum prob (not strictly prob as it haven't gone
        # through softmax yet so it doesn't sum to 1, but result is the same
        # for argmax).
        i = np.argmax(predt[r])
        out[r] = i
    return out


def plot_history(custom_results, native_results):
    fig, axs = plt.subplots(2, 1)
    ax0 = axs[0]
    ax1 = axs[1]

    x = np.arange(0, kRounds, 1)
    ax0.plot(x, custom_results['train']['merror'], label='Focal loss objective')
    ax0.legend()
    ax1.plot(x, native_results['train']['merror'], label='multi:softmax')
    ax1.legend()

    plt.show()


def main(args):
    custom_results = {}
    # Use our custom objective function
    booster_custom = xgb.train({'num_class': kClasses},
                               m,
                               num_boost_round=kRounds,
                               obj=log_focal_loss_obj,
                               evals_result=custom_results,
                               evals=[(m, 'train')])

    #predt_custom = predict(booster_custom, m)
    predt_custom = booster_custom.predict(m)
    print(predt_custom[0:10])
    predt_raw = booster_custom.predict(m, output_margin=True)
    predt_score=[softmax(predt_raw[i]) for i in range(len(predt_raw))]
    print(predt_score[0:10])

    native_results = {}

    booster_native = xgb.train({'num_class': kClasses},
                               m,
                               num_boost_round=kRounds,
                               evals_result=native_results,
                               evals=[(m, 'train')])
    predt_native = booster_native.predict(m)
    print(predt_native[0:10])


    if args.plot != 0:
        plot_history(custom_results, native_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for custom focal loss objective function demo.')
    parser.add_argument(
        '--plot',
        type=int,
        default=1,
        help='Set to 0 to disable plotting the evaluation history.')
    args = parser.parse_args()
    main(args)


