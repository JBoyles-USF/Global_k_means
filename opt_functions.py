# module opt_functions
#
# using Clustering
# using Printf
# using JuMP
# using Ipopt, CPLEX, Gurobi#, SCIP
# using Random
# using InteractiveUtils
#
# export obj_assign, local_OPT, global_OPT_base, global_OPT3_LD
#
# time_lapse = 900 # 15 mins
# obbt_time = 180 # 1 mins

import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Time limit for solvers (seconds)
time_lapse = 900  # 15 mins
# Optimization-Based Bound Tightening (OBBT) time — currently unused
# obbt_time = 180  # 1 min




############# auxilary functions #############

# function obj_assign(centers, X)
#     d, n = size(X)   	 
#     k = size(centers, 2)	 
#     dmat = zeros(k, n)
#     for i=1:n
#     	for j = 1:k
#     	    dmat[j,i] = sum((X[:,i] .- centers[:,j]).^2) # dmat[j,i] is the distance from point i to center j
# 	    end    
#     end
# 
#     assign = Vector{Int}(undef, n)
#     costs = Vector{Float64}(undef, n)
#     for j = 1:n
#         c, a = findmin(dmat[:, j]) # find the closest cluster that point j belongs to
# 	    assign[j] = a # a is the cluster label of point j
#         costs[j] = c # c is the distance of point j to center of cluster a
#     end    
#     return sum(costs), assign # sum costs is the total sse, assign is the current clustering assignment

import numpy as np

def obj_assign(centers, X):
    """
    Compute clustering assignment and total sum of squared errors (SSE).
    
    centers: np.array of shape (d, k)
    X: np.array of shape (d, n)
    
    Returns:
        total_sse: float
        assign: np.array of shape (n,) with cluster indices
    """
    # d, n = size(X)
    d, n = X.shape
    # k = size(centers, 2)
    k = centers.shape[1]
    # dmat = zeros(k, n)
    dmat = np.zeros((k, n))
    
    # for i=1:n
    for i in range(n):
        # for j = 1:k
        for j in range(k):
            # dmat[j,i] = sum((X[:,i] .- centers[:,j]).^2) # distance from point i to center j
            dmat[j, i] = np.sum((X[:, i] - centers[:, j]) ** 2)
    
    # assign = Vector{Int}(undef, n)
    # costs = Vector{Float64}(undef, n)
    assign = np.zeros(n, dtype=int)
    costs = np.zeros(n, dtype=float)
    
    # for j = 1:n
    for j in range(n):
        # c, a = findmin(dmat[:, j]) # find closest cluster for point j
        a = np.argmin(dmat[:, j])
        c = dmat[a, j]
        # assign[j] = a # cluster label
        assign[j] = a
        # costs[j] = c # distance to cluster center
        costs[j] = c
    
    # return sum(costs), assign
    return np.sum(costs), assign






# function init_bound(X, d, k, lower=nothing, upper=nothing)
#     lower_data = Vector{Float64}(undef, d)
#     upper_data = Vector{Float64}(undef, d)
import numpy as np

def init_bound(X, d, k, lower=None, upper=None):
    """
    Initialize lower and upper bounds for cluster centers.

    X: np.array of shape (d, n)
    d: int, number of dimensions
    k: int, number of clusters
    lower, upper: optional np.array of shape (d, k) specifying bounds

    Returns:
        lower, upper: np.array of shape (d, k)
    """
    # for i = 1:d # get the feasible region of center
    lower_data = np.zeros(d)
    upper_data = np.zeros(d)
    for i in range(d):
        # lower_data[i] = minimum(X[i,:]) # i is the row and is the dimension 
        lower_data[i] = np.min(X[i, :])
        # upper_data[i] = maximum(X[i,:])
        upper_data[i] = np.max(X[i, :])
    
    # lower_data = repeat(lower_data, 1, k) # first arg repeat on row, second repeat on col
    # upper_data = repeat(upper_data, 1, k)
    lower_data = np.tile(lower_data.reshape(d, 1), (1, k))
    upper_data = np.tile(upper_data.reshape(d, 1), (1, k))
    
    # if lower === nothing
    if lower is None:
        # lower = lower_data
        # upper = upper_data
        lower = lower_data
        upper = upper_data
    else:
        # lower = min.(upper.-1e-4, max.(lower, lower_data))
        lower = np.minimum(upper - 1e-4, np.maximum(lower, lower_data))
        # upper = max.(lower.+1e-4, min.(upper, upper_data))
        upper = np.maximum(lower + 1e-4, np.minimum(upper, upper_data))
    
    # return lower, upper
    return lower, upper





# function max_dist(X, d, k, n, lower, upper)
#     dmat_max = zeros(k,n)
import numpy as np

def max_dist(X, d, k, n, lower, upper):
    """
    Compute the maximum squared distance from each point to the bounds of each cluster center.

    X: np.array of shape (d, n)
    d: int, number of dimensions
    k: int, number of clusters
    n: int, number of points
    lower, upper: np.array of shape (d, k)

    Returns:
        dmat_max: np.array of shape (k, n)
    """
    # for j = 1:n
    dmat_max = np.zeros((k, n))
    for j in range(n):
        # for i = 1:k
        for i in range(k):
            # max_distance = 0
            max_distance = 0
            # for t = 1:d
            for t in range(d):
                # max_distance += max((X[t,j]-lower[t,i])^2, (X[t,j]-upper[t,i])^2)
                max_distance += max((X[t, j] - lower[t, i])**2, (X[t, j] - upper[t, i])**2)
            # dmat_max[i,j] = max_distance
            dmat_max[i, j] = max_distance
    # return dmat_max
    return dmat_max




# ############# local optimization function (Ipopt) ############# 
# function local_OPT(X, k, lower=nothing, upper=nothing)
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def local_OPT(X, k, lower=None, upper=None):
    # d, n = size(X)
    d, n = X.shape
    # lower, upper = init_bound(X, d, k, lower, upper)
    lower, upper = init_bound(X, d, k, lower, upper)

    # m = Model(Ipopt.Optimizer);
    m = gp.Model("local_OPT")  # Using Gurobi instead of Ipopt
    # set_optimizer_attribute(m, "print_level", 0);
    m.Params.OutputFlag = 0  # suppress Gurobi output

    # @variable(m, lower[i,j] <= centers[i in 1:d, j in 1:k] <= upper[i,j], start=rand());
    centers = {}
    for i in range(d):
        for j in range(k):
            centers[i, j] = m.addVar(lb=lower[i, j], ub=upper[i, j], name=f"c_{i}_{j}")

    # @constraint(m, [j in 1:k-1], centers[1,j]<= centers[1,j+1])
    for j in range(k-1):
        m.addConstr(centers[0, j] <= centers[0, j+1], name=f"sort_{j}")

    # @variable(m, dmat[1:k,1:n], start=rand());
    dmat = {}
    for i in range(k):
        for j in range(n):
            dmat[i, j] = m.addVar(lb=0, name=f"d_{i}_{j}")

    # @constraint(m, [i in 1:k, j in 1:n], dmat[i,j] >= sum((X[t,j] - centers[t,i])^2 for t in 1:d ));
    for i in range(k):
        for j in range(n):
            expr = sum((X[t, j] - centers[t, i])**2 for t in range(d))
            m.addConstr(dmat[i, j] >= expr, name=f"dmat_constr_{i}_{j}")

    # @variable(m, 0<=lambda[1:k,1:n]<=1,	start=rand());
    lambda_var = {}
    for i in range(k):
        for j in range(n):
            lambda_var[i, j] = m.addVar(lb=0, ub=1, name=f"lambda_{i}_{j}")

    # @constraint(m, [j in 1:n], sum(lambda[i,j] for i in 1:k) == 1);
    for j in range(n):
        m.addConstr(gp.quicksum(lambda_var[i, j] for i in range(k)) == 1, name=f"lambda_sum_{j}")

    # @variable(m, costs[1:n], start=rand());
    costs = {}
    for j in range(n):
        costs[j] = m.addVar(lb=0, name=f"cost_{j}")

    # @constraint(m, [j in 1:n], costs[j] == sum(lambda[i,j]*dmat[i,j] for i in 1:k));
    for j in range(n):
        m.addConstr(costs[j] == gp.quicksum(lambda_var[i, j] * dmat[i, j] for i in range(k)), name=f"cost_constr_{j}")

    # @objective(m, Min, sum(costs[j] for j in 1:n));
    m.setObjective(gp.quicksum(costs[j] for j in range(n)), GRB.MINIMIZE)

    # optimize!(m);
    m.optimize()

    # centers = value.(centers)
    centers_val = np.zeros((d, k))
    for i in range(d):
        for j in range(k):
            centers_val[i, j] = centers[i, j].X

    # objv, assign = obj_assign(centers, X)
    from copy import deepcopy
    objv, assign = obj_assign(centers_val, X)

    # #objv = getobjectivevalue(m)
    # return centers, assign, objv
    return centers_val, assign, objv





# ############# global optimization solvers #############
# pure cplex solvers
# function global_OPT_base(X, k, lower=nothing, upper=nothing, mute=false)
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def global_OPT_base(X, k, lower=None, upper=None, mute=False, time_lapse=900):
    # d, n = size(X)
    d, n = X.shape
    # lower, upper = init_bound(X, d, k, lower, upper)
    lower, upper = init_bound(X, d, k, lower, upper)
    # dmat_max = max_dist(X, d, k, n, lower, upper)
    dmat_max = max_dist(X, d, k, n, lower, upper)

    # m = Model(CPLEX.Optimizer);
    m = gp.Model("global_OPT_base")  # using Gurobi instead of CPLEX
    # if mute
    #     set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
    if mute:
        m.Params.OutputFlag = 0  # suppress output

    # set_optimizer_attribute(m, "CPX_PARAM_THREADS",1)
    m.Params.Threads = 1
    # set_optimizer_attribute(m, "CPX_PARAM_TILIM", time_lapse*16) # maximum runtime limit
    m.Params.TimeLimit = time_lapse * 16
    # set_optimizer_attribute(m, "CPX_PARAM_MIQCPSTRAT", 0) # 0 for qcp relax
    # # set_optimizer_attribute(m, "MIQCPMethod", 1) # 0 for qcp relax and 1 for lp oa relax, -1 for auto This is for Gurobi

    # @variable(m, lower[t,i] <= centers[t in 1:d, i in 1:k] <= upper[t,i], start=rand());
    centers = {}
    for t in range(d):
        for i in range(k):
            centers[t, i] = m.addVar(lb=lower[t, i], ub=upper[t, i], name=f"center_{t}_{i}")

    # @constraint(m, [j in 1:k-1], centers[1,j]<= centers[1,j+1])
    for j in range(k-1):
        m.addConstr(centers[0, j] <= centers[0, j+1], name=f"sort_{j}")

    # @variable(m, 0<=dmat[i in 1:k, j in 1:n]<=dmat_max[i,j], start=rand());
    dmat = {}
    for i in range(k):
        for j in range(n):
            dmat[i, j] = m.addVar(lb=0, ub=dmat_max[i, j], name=f"dmat_{i}_{j}")

    # @constraint(m, [i in 1:k, j in 1:n], dmat[i,j] >= sum((X[t,j] - centers[t,i])^2 for t in 1:d ));
    for i in range(k):
        for j in range(n):
            expr = sum((X[t, j] - centers[t, i])**2 for t in range(d))
            m.addConstr(dmat[i, j] >= expr, name=f"dmat_constr_{i}_{j}")

    # @variable(m, lambda[1:k, 1:n], Bin)
    lambda_var = {}
    for i in range(k):
        for j in range(n):
            lambda_var[i, j] = m.addVar(vtype=GRB.BINARY, name=f"lambda_{i}_{j}")

    # @constraint(m, [j in 1:n], sum(lambda[i,j] for i in 1:k) == 1);
    for j in range(n):
        m.addConstr(gp.quicksum(lambda_var[i, j] for i in range(k)) == 1, name=f"assign_sum_{j}")

    # @variable(m, costs[1:n], start=rand());
    costs = {}
    for j in range(n):
        costs[j] = m.addVar(lb=0, name=f"cost_{j}")

    # @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] >= -dmat_max[i,j]*(1-lambda[i,j]))
    # @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] <= dmat_max[i,j]*(1-lambda[i,j]))
    for i in range(k):
        for j in range(n):
            m.addConstr(costs[j] - dmat[i, j] >= -dmat_max[i, j] * (1 - lambda_var[i, j]), name=f"cost_lb_{i}_{j}")
            m.addConstr(costs[j] - dmat[i, j] <= dmat_max[i, j] * (1 - lambda_var[i, j]), name=f"cost_ub_{i}_{j}")

    # @objective(m, Min, sum(costs[j] for j in 1:n));
    m.setObjective(gp.quicksum(costs[j] for j in range(n)), GRB.MINIMIZE)

    # optimize!(m);
    m.optimize()

    # centers = value.(centers)
    centers_val = np.zeros((d, k))
    for t in range(d):
        for i in range(k):
            centers_val[t, i] = centers[t, i].X

    # node = node_count(m)
    node = m.NodeCount
    # gap = relative_gap(m) # get the relative gap for cplex solver
    gap = m.MIPGap
    # objv, ~ = obj_assign(centers, X) # here the objv should be a lower bound of CPLEX
    objv, assign = obj_assign(centers_val, X)

    # return centers, objv, node, gap
    return centers_val, objv, node, gap








# reduced bb subproblem solvers with largranian decomposition
# here the labmda is the largrange multiplier
#function global_OPT3_LD(X, k, lambda, ctr_init, w_sos=nothing, lower=nothing, upper=nothing, mute=false, solver="CPLEX")
#    d, n = size(X)
#    lower, upper = init_bound(X, d, k, lower, upper)
#    dmat_max = max_dist(X, d, k, n, lower, upper)
#    
#    #w_bin = rlt.centers # weight of the binary variables
#    if solver=="CPLEX"
#        m = Model(CPLEX.Optimizer);
#        if mute
#            set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
#        end
#        set_optimizer_attribute(m, "CPX_PARAM_THREADS",1)
#        set_optimizer_attribute(m, "CPX_PARAM_TILIM", time_lapse) # maximum runtime limit is 1 hours
#        # here the gap should always < mingap of BB, e.g. if mingap = 0.1%, then gap here should be < 0.1%, the default is 0.01%
#        # set_optimizer_attribute(m, "CPX_PARAM_EPGAP", 0.05) 
#    else # solver is Gurobi
#        m = Model(Gurobi.Optimizer);
#        if mute
#            set_optimizer_attribute(m, "OutputFlag", 0)
#        end
#        set_optimizer_attribute(m, "Threads",1)
#        set_optimizer_attribute(m, "TimeLimit", time_lapse) # maximum runtime limit is 1 hours
#        # set_optimizer_attribute(m, "PreMIQCPForm", 0) # improve the speed of bb process
#        # here the gap should always < mingap of BB, e.g. if mingap = 0.1%, then gap here should be < 0.1%, the default is 0.01%
#        # set_optimizer_attribute(m, "MIPGap", 0.05) 
#    end
#    @variable(m, lower[t,i] <= centers[t in 1:d, i in 1:k] <= upper[t,i], start=ctr_init[t,i]);
#    @constraint(m, [j in 1:k-1], centers[1,j]<= centers[1,j+1])
#    @variable(m, 0<=dmat[i in 1:k, j in 1:n]<=dmat_max[i,j], start=rand());
#    @constraint(m, [i in 1:k, j in 1:n], dmat[i,j] >= sum((X[t,j] - centers[t,i])^2 for t in 1:d ));
#
#    #@constraint(m, [i in 1:k, j in 1:n], [dmat[i,j] X[:,j]-centers[:,i]] in SecondOrderCone())
#    #@constraint(m, [i in 1:k, j in 1:n], [dmat[i,j]; X[:,j]-centers[:,i]] in SecondOrderCone())
#    @variable(m, b[1:k, 1:n], Bin)
#    @constraint(m, [j in 1:n], sum(b[i,j] for i in 1:k) == 1);
#    @constraint(m, [j in 1:n], b[:,j] in MOI.SOS1(w_sos[:,j])) #SOS1 constraint
#
#    @variable(m, costs[1:n]>=0, start=rand());
#    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] >= -dmat_max[i,j]*(1-b[i,j]))
#    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] <= dmat_max[i,j]*(1-b[i,j]))
#
#    @objective(m, Min, sum(costs[j] for j in 1:n)+
#                sum((lambda[:,i,2]-lambda[:,i,1])'*centers[:,i] for i in 1:k));
#    optimize!(m);
#
#
#    centers = value.(centers)
#    objv = objective_bound(m)
#    return centers, objv
#end
#
#end 
