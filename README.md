# Electric-vehicle-charging-and-Routing-problem
Implemented ALNS and solved a simplified version of electic vehicle charging and routing problem with nonlinear charging functions
## Problem Statement
Unlike the standard VRP, E-VRP involves both sequencing decisions as well as charging decisions. Here I simplify E-VRP-NL problem by assuming that every customer can always be served by no more than one visit to a charging station. And solve it with ALNS algorithm.
## Construction Heuristic 
### Initial route: 
Since one customer can only be visited once, I randomly shuffle the customers and use the list as initial route, assuming that the sequence of the list will be the sequence of visiting the customers.
### Split route into sub routes：
In this split function I assume that with full battery, a vehicle will be able to move to any node.
Initialize a vehicle v to store the current sub route
### Destroy methods
From the split route function, we get feasible solution, on top of that, I implement destroy and repair to find optimal solution.
- Worst destroy: 
- Randomly select a sub-route using random state. 
- For each customer node C in this route, calculate the sum of distance between itself and the customers which are one step before and after it.
-	Find the customer with largest distance to its neighbours.
-	Destroy this customer node
- Random destroy:
- randomly select a sub-route using random state.
- randomly destroy a node in the subroute
### Repair methods 
![image](https://user-images.githubusercontent.com/62283777/161905133-d74117e6-defb-4687-812c-62c72738922b.png)
- greedy repair:
- Find all customer unvisited and insert them to the existing route one by one.
- The existing route is the list of customers visited.
- Calculate the sum of distance between unvisited customers and pairs of subsequent visited customers, insert the unvisited customer into the pair with smallest distance
## Weights adjustment strategy
- Choose the hill climbing criterion, only accepts progressively better solutions, discarding those that result in a worse objective value, 
- omegas = [3,1,0,0.5] highest weight to best candidate, 0 weight to infeasible candidate, 0.5 weight to accepted candidate and 0 weight to infeasible candidate
- lambda_ = 0.8, more sensible to changes



