###############################################################
# TW-GRILS Balanced Multi-Day + MILP-style Goal Programming Z
# - Parallel processing
# - Daily constraints
# - Travel time +5 minutes, travel cost +50 Baht (MILP)
# - Balanced-day scoring to avoid empty days
###############################################################

import pandas as pd
import numpy as np
import random
import concurrent.futures


class TWGRILSOptimizer:

    ###################################################################
    # INITIALIZATION
    ###################################################################
    def __init__(self,
                 excel_path,
                 alpha=0.7,
                 beta=1.0,
                 gamma=0.3,
                 T_start=8.0,
                 T_end=18.0,
                 K=5,
                 rng_seed=None,
                 min_safety=None,
                 max_walking=None,
                 min_pathway=None,
                 max_total_budget=None,
                 max_travel_time_minutes=None,
                 num_workers=None):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.T_start = T_start
        self.T_end = T_end
        self.K = K
        self.num_workers = num_workers

        # constraints
        self.min_safety = min_safety
        self.max_walking = max_walking
        self.min_pathway = min_pathway
        self.max_total_budget = max_total_budget
        self.max_travel_time_minutes = max_travel_time_minutes

        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        self._load_data(excel_path)
        self._precompute_bounds()

        # ----------------------------------------
        # GOAL Programming Targets (MILP style)
        # ----------------------------------------
        self.goal_S  = 50
        self.goal_SF = 50
        self.goal_CL = 50
        self.goal_AC = 50
        self.goal_CU = 50
        self.goal_PH = 50
        self.goal_SC = 50
        self.goal_WK = 2500
        self.goal_C  = 1000
        self.goal_T  = 200

        # weights (MILP = 0.1 ทุก objective)
        self.w_S  = 0.1
        self.w_SF = 0.1
        self.w_CL = 0.1
        self.w_AC = 0.1
        self.w_CU = 0.1
        self.w_PH = 0.1
        self.w_SC = 0.1
        self.w_WK = 0.1
        self.w_C  = 0.1
        self.w_T  = 0.1


    ###################################################################
    # LOAD DATA
    ###################################################################
    def _load_data(self, excel_path):
        df_input = pd.read_excel(excel_path, sheet_name="Input")
        df_dist  = pd.read_excel(excel_path, sheet_name="Distance", index_col=0)
        df_time  = pd.read_excel(excel_path, sheet_name="Time", index_col=0)
        df_cost  = pd.read_excel(excel_path, sheet_name="Cost", index_col=0)

        self.locations = df_input["Locations"].tolist()
        self.N = len(self.locations) - 1  # node 0 = depot

        self.S  = df_input["S"].values
        self.R  = df_input["R"].values
        self.a  = df_input["TW_min"].values
        self.b  = df_input["TW_max"].values

        self.Safety        = df_input["Safety"].values
        self.Comfort       = df_input["Comf"].values
        self.Accessibility = df_input["Acess"].values
        self.Culture       = df_input["Culture"].values
        self.Pathway       = df_input["Pathway"].values
        self.Scenic        = df_input["Scen"].values
        self.Walking       = df_input["Walking"].values
        self.Fee           = df_input["Fee"].values

        self.dist = df_dist.values.astype(float)
        self.t    = df_time.values.astype(float) / 60.0
        self.cost = df_cost.values.astype(float)

        self.big_dist = 9999
        self.big_time = 9999
        self.big_cost = 9999


    ###################################################################
    # PRECOMPUTE
    ###################################################################
    def _precompute_bounds(self):
        self.Score_max = max(1e-6, self.S.sum())
        valid = self.dist[self.dist < self.big_dist]
        self.Dist_max = max(1e-6, valid.mean() * (self.N + 1)) if valid.size else 1.0


    ###################################################################
    # GOAL PROGRAMMING Z (MILP)
    ###################################################################
    def compute_Z(self, m):
        EPS = 1e-6

        S  = m["Satisfaction"]
        SF = m["Safety"]
        CL = m["Comfort"]
        AC = m["Accessibility"]
        CU = m["Culture"]
        PH = m["Pathway"]
        SC = m["Scenic"]
        WK = m["Walking"]
        C  = m["TotalTravelCost"]
        T  = m["TotalTravelTime"]

        dev_S  = max(self.goal_S  - S , 0)
        dev_SF = max(self.goal_SF - SF, 0)
        dev_CL = max(self.goal_CL - CL, 0)
        dev_AC = max(self.goal_AC - AC, 0)
        dev_CU = max(self.goal_CU - CU, 0)
        dev_PH = max(self.goal_PH - PH, 0)
        dev_SC = max(self.goal_SC - SC, 0)

        dev_WK = max(WK - self.goal_WK, 0)
        dev_C  = max(C  - self.goal_C , 0)
        dev_T  = max(T  - self.goal_T , 0)

        Z = (
            self.w_S  * dev_S  / max(self.goal_S , EPS) +
            self.w_SF * dev_SF / max(self.goal_SF, EPS) +
            self.w_CL * dev_CL / max(self.goal_CL, EPS) +
            self.w_AC * dev_AC / max(self.goal_AC, EPS) +
            self.w_CU * dev_CU / max(self.goal_CU, EPS) +
            self.w_PH * dev_PH / max(self.goal_PH, EPS) +
            self.w_SC * dev_SC / max(self.goal_SC, EPS) +
            self.w_WK * dev_WK / max(self.goal_WK, EPS) +
            self.w_C  * dev_C  / max(self.goal_C , EPS) +
            self.w_T  * dev_T  / max(self.goal_T , EPS)
        )
        return Z


    ###################################################################
    # EVALUATE ROUTE (with +5 min and +50 cost)
    ###################################################################
    def evaluate(self, route):

        time = self.T_start
        score = 0
        dist_total = 0
        cost_total = 0
        travel_cost_total = 0
        feasible = True

        metrics = {
            "Satisfaction":0, "Safety":0, "Comfort":0, "Accessibility":0, "Culture":0,
            "Pathway":0, "Scenic":0, "Walking":0,
            "TotalTravelTime":0, "TotalTravelCost":0
        }

        for k in range(len(route)-1):
            i, j = route[k], route[k+1]

            bd = self.dist[i,j]
            bt = self.t[i,j]
            bc = self.cost[i,j]

            if bd>=self.big_dist or bt>=self.big_time or bc>=self.big_cost:
                feasible=False; break

            # MILP adjustments:
            d_ij = bd
            t_ij = bt + 5/60
            c_ij = bc + 50

            metrics["TotalTravelTime"] += t_ij * 60
            travel_cost_total += c_ij
            time += t_ij

            if j!=0:
                if self.min_safety and self.Safety[j] < self.min_safety: feasible=False; break
                if self.min_pathway and self.Pathway[j] < self.min_pathway: feasible=False; break

                if time < self.a[j]:
                    time = self.a[j]
                if time > self.b[j]:
                    feasible=False; break

                time += self.R[j]

                score += self.S[j]
                cost_total += self.Fee[j]

                metrics["Satisfaction"] += self.S[j]
                metrics["Safety"]       += self.Safety[j]
                metrics["Comfort"]      += self.Comfort[j]
                metrics["Accessibility"]+= self.Accessibility[j]
                metrics["Culture"]      += self.Culture[j]
                metrics["Pathway"]      += self.Pathway[j]
                metrics["Scenic"]       += self.Scenic[j]
                metrics["Walking"]      += self.Walking[j]

                if self.max_walking and metrics["Walking"] > self.max_walking:
                    feasible=False; break

            dist_total += d_ij
            cost_total += c_ij

            if self.max_travel_time_minutes and metrics["TotalTravelTime"] > self.max_travel_time_minutes:
                feasible=False; break
            if time > self.T_end:
                feasible=False; break

        if self.max_total_budget and cost_total > self.max_total_budget:
            feasible=False

        metrics["TotalTravelCost"] = travel_cost_total

        if not feasible:
            return -1e9, dist_total, score, cost_total, False, metrics

        F = (
            self.alpha * (score / self.Score_max)
            - (1-self.alpha) * (dist_total / self.Dist_max)
        )
        return F, dist_total, score, cost_total, True, metrics



    ###################################################################
    # BALANCED GREEDY (สำคัญที่สุด)
    ###################################################################
    def greedy_randomized_construction_balanced(self, day_index, num_days, allowed_nodes):

        # penalty: วันแรกโดนเยอะ วันสุดท้ายไม่โดนเลย
        day_penalty = 1 - (day_index / num_days)

        route=[0]
        visited={0}
        current=0
        current_time=self.T_start
        cost_total=0
        walking_total=0
        travel_minutes=0

        while True:
            candidates=[]

            for j in range(1, self.N+1):
                if j in visited: continue
                if allowed_nodes and j not in allowed_nodes: continue

                bd = self.dist[current,j]
                bt = self.t[current,j]
                bc = self.cost[current,j]

                if bd>=self.big_dist or bt>=self.big_time or bc>=self.big_cost:
                    continue

                d_ij = bd
                t_ij = bt + 5/60
                c_ij = bc + 50

                arrive = current_time + t_ij
                arrive2= max(arrive, self.a[j])
                depart = arrive2 + self.R[j]

                if depart > self.b[j] or depart > self.T_end: continue

                new_cost = cost_total + c_ij + self.Fee[j]
                if self.max_total_budget and new_cost > self.max_total_budget:
                    continue

                new_walk = walking_total + self.Walking[j]
                if self.max_walking and new_walk > self.max_walking:
                    continue

                new_travel = travel_minutes + t_ij*60
                if self.max_travel_time_minutes and new_travel > self.max_travel_time_minutes:
                    continue

                score_inc = self.S[j]

                value = score_inc / (d_ij**self.beta)
                value -= day_penalty     # ★ KEY OF BALANCING ★

                candidates.append((j,value,depart,new_cost,new_walk,new_travel))

            if not candidates:
                break

            candidates.sort(key=lambda x:x[1], reverse=True)
            rcl = candidates[:min(self.K, len(candidates))]
            j,v,depart,new_cost,new_walk,new_travel = random.choice(rcl)

            route.append(j)
            visited.add(j)
            current=j
            current_time=depart
            cost_total=new_cost
            walking_total=new_walk
            travel_minutes=new_travel

        route.append(0)
        return route



    ###################################################################
    # LOCAL SEARCH / ILS
    ###################################################################
    @staticmethod
    def two_opt(route,i,k):
        return route[:i] + route[i:k+1][::-1] + route[k+1:]

    def local_search(self, route, max_no=40):
        best=route[:]
        best_F, *_ = self.evaluate(best)

        counter=0
        n=len(best)

        while counter<max_no:
            improved=False
            for i in range(1,n-2):
                for k in range(i+1,n-1):
                    newr=self.two_opt(best,i,k)
                    F, *_ = self.evaluate(newr)
                    if F > best_F:
                        best=newr
                        best_F=F
                        improved=True
                        break
                if improved: break

            counter = 0 if improved else counter+1
        return best


    def iterated_local_search(self, route, max_ils):
        current=self.local_search(route)
        best=current[:]
        best_F, *_ = self.evaluate(best)

        for _ in range(max_ils):
            pert=self._perturb(current)
            cand=self.local_search(pert)
            F, *_ = self.evaluate(cand)
            if F > best_F:
                best=cand
                best_F=F
                current=cand
            else:
                current=best

        return best


    def _perturb(self, route, s=2):
        if len(route)<=4: return route[:]
        mid=route[1:-1]
        for _ in range(s):
            i,j = random.sample(range(len(mid)),2)
            mid[i],mid[j]=mid[j],mid[i]
        return [0]+mid+[0]


    ###################################################################
    # SINGLE-DAY WITH BALANCED GREEDY
    ###################################################################
    def _run_one_iteration_balanced(self, allowed, max_ils, day_index, num_days):

        s0 = self.greedy_randomized_construction_balanced(day_index, num_days, allowed)
        s  = self.iterated_local_search(s0, max_ils)
        F, dist_total, score, cost_total, feas, metrics = self.evaluate(s)

        if not feas:
            return None

        return F, s, (dist_total, score, cost_total, metrics)


    def optimize_single_day_balanced(self, allowed, day_index, num_days,
                                     max_iter=30, max_ils=20):

        best_route=None
        best_val=-1e9
        best_stats=None

        workers = self.num_workers or None

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures=[
                ex.submit(self._run_one_iteration_balanced,
                          allowed, max_ils, day_index, num_days)
                for _ in range(max_iter)
            ]

            for f in concurrent.futures.as_completed(futures):
                r=f.result()
                if r is None: continue
                F,route,stats=r
                if F>best_val:
                    best_val=F
                    best_route=route
                    best_stats=stats

        return best_route, best_val, best_stats



    ###################################################################
    # MULTI-DAY with BALANCING
    ###################################################################
    def optimize_multi_day(self, num_days, max_iter=30, max_ils=20):

        remaining=set(range(1,self.N+1))
        day_routes=[]
        day_values=[]
        day_stats=[]

        for d in range(num_days):
            day_index=d+1

            route,val,stats = self.optimize_single_day_balanced(
                allowed=remaining,
                day_index=day_index,
                num_days=num_days,
                max_iter=max_iter,
                max_ils=max_ils
            )

            if route is None or len(route)<=2:
                empty={
                    "Satisfaction":0,"Safety":0,"Comfort":0,"Accessibility":0,
                    "Culture":0,"Pathway":0,"Scenic":0,"Walking":0,
                    "TotalTravelTime":0,"TotalTravelCost":0
                }
                day_routes.append([0,0])
                day_values.append(-1e9)
                day_stats.append((0,0,0,empty))
                continue

            day_routes.append(route)
            day_values.append(val)
            day_stats.append(stats)

            visited=set(route)
            visited.discard(0)
            remaining -= visited

        # summary whole trip
        total_metrics={
            "Satisfaction":0,"Safety":0,"Comfort":0,"Accessibility":0,
            "Culture":0,"Pathway":0,"Scenic":0,"Walking":0,
            "TotalTravelTime":0,"TotalTravelCost":0
        }
        total_dist=0
        total_score=0
        total_cost=0

        for dist_total,score,cost_total,m in day_stats:
            total_dist += dist_total
            total_score+= score
            total_cost += cost_total
            for k in total_metrics:
                total_metrics[k]+=m[k]

        return day_routes, day_values, day_stats, (total_dist,total_score,total_cost,total_metrics)



#######################################################################
# PRINTING HELPERS
#######################################################################
def print_day(opt, route, val, stats, day):
    d,s,c,m=stats
    print(f"\n=========== DAY {day} ===========")
    print("Route idx :", route)
    print("Locations :", [opt.locations[i] for i in route])
    print(f"Distance  : {d:.2f}")
    print(f"Score(S)  : {s:.2f}")
    print(f"Cost      : {c:.2f}")
    print("\n---- KPIs ----")
    for k,v in m.items():
        print(f"{k:18s}: {v:.2f}")
    print(f"\nDay Objective Z = {opt.compute_Z(m):.6f}")


def print_trip(opt, routes, vals, stats, total_stats, num_days):
    for d,(r,v,s) in enumerate(zip(routes,vals,stats),start=1):
        print_day(opt,r,v,s,d)

    td,ts,tc,tm = total_stats
    print("\n================ TRIP SUMMARY ================")
    print(f"Total Distance : {td:.2f}")
    print(f"Total Score    : {ts:.2f}")
    print(f"Total Cost     : {tc:.2f}")
    print("\n---- Total KPIs ----")
    for k,v in tm.items():
        print(f"{k:18s}: {v:.2f}")
    print(f"\nTOTAL TRIP Z = {opt.compute_Z(tm):.6f}")



#######################################################################
# MAIN
#######################################################################
if __name__=="__main__":

    excel_path="Input Data.xlsx"

    NUM_DAYS = 2        # ← เปลี่ยนเป็น 2 หรือ 3 วันได้ทันที
    MAX_ITER = 12
    MAX_ILS  = 20

    opt = TWGRILSOptimizer(
        excel_path=excel_path,
        rng_seed=42,
        min_safety=3,
        min_pathway=3,
        max_walking=2500,
        max_total_budget=1000,
        max_travel_time_minutes=200,
        num_workers=None
    )

    routes,vals,stats,total = opt.optimize_multi_day(
        NUM_DAYS, max_iter=MAX_ITER, max_ils=MAX_ILS
    )

    print_trip(opt, routes, vals, stats, total, NUM_DAYS)