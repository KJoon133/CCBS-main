import time
import heapq
import random
import numpy as np
# from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
import sys
from core.func.a_star import a_star, compute_heuristics_costmap, get_sum_of_cost, compute_heuristics, get_location, compute_heuristics_uniform_costmap
# from pea_star import pea_star


from core.func.a_star_class import A_Star
from core.func.pea_star_class import PEA_Star

PEA_STAR = PEA_Star

def detect_collision(path1, path2):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.
    t_range = max(len(path1),len(path2))
    for t in range(t_range):
        loc_c1 =get_location(path1,t)
        loc_c2 = get_location(path2,t)
        loc1 = get_location(path1,t+1)
        loc2 = get_location(path2,t+1)
        if loc1 == loc2:
            return [loc1],t
        if[loc_c1,loc1] ==[loc2,loc_c2]:
            return [loc2,loc_c2],t
        
       
    return None
    # pass


def detect_collisions(paths):
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.
    collisions =[]
    for i in range(len(paths)-1):
        for j in range(i+1,len(paths)):
            if detect_collision(paths[i],paths[j]) !=None:
                position,t = detect_collision(paths[i],paths[j])
                collisions.append({'a1':i,
                                'a2':j,
                                'loc':position,
                                'timestep':t+1})
    return collisions


def standard_splitting(collision):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep
    constraints = []
    if len(collision['loc'])==1:
        constraints.append({'agent':collision['a1'],
                            'loc':collision['loc'],
                            'timestep':collision['timestep'],
                            'positive':False
                            })
        constraints.append({'agent':collision['a2'],
                            'loc':collision['loc'],
                            'timestep':collision['timestep'],
                            'positive':False
                            })
    else:
        constraints.append({'agent':collision['a1'],
                            'loc':[collision['loc'][0],collision['loc'][1]],
                            'timestep':collision['timestep'],
                            'positive':False
                            })
        constraints.append({'agent':collision['a2'],
                            'loc':[collision['loc'][1],collision['loc'][0]],
                            'timestep':collision['timestep'],
                            'positive':False
                            })
    return constraints


def disjoint_splitting(collision):
    ##############################
    # Task 4.1: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint enforces one agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the same agent to be at the
    #                            same location at the timestep.
    #           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the same agent to traverse the
    #                          specified edge at the specified timestep
    #           Choose the agent randomly
    constraints = []
    agent = random.randint(0,1)
    a = 'a'+str(agent +1)
    if len(collision['loc'])==1:
        constraints.append({'agent':collision[a],
                            'loc':collision['loc'],
                            'timestep':collision['timestep'],
                            'positive':True
                            })
        constraints.append({'agent':collision[a],
                            'loc':collision['loc'],
                            'timestep':collision['timestep'],
                            'positive':False
                            })
    else:
        if agent ==0:
            constraints.append({'agent':collision[a],
                                'loc':[collision['loc'][0],collision['loc'][1]],
                                'timestep':collision['timestep'],
                                'positive':True
                                })
            constraints.append({'agent':collision[a],
                                'loc':[collision['loc'][0],collision['loc'][1]],
                                'timestep':collision['timestep'],
                                'positive':False
                                })
        else:
            constraints.append({'agent':collision[a],
                                'loc':[collision['loc'][1],collision['loc'][0]],
                                'timestep':collision['timestep'],
                                'positive':True
                                })
            constraints.append({'agent':collision[a],
                                'loc':[collision['loc'][1],collision['loc'][0]],
                                'timestep':collision['timestep'],
                                'positive':False
                                })
    return constraints


    # pass

def paths_violate_constraint(constraint, paths):
    assert constraint['positive'] is True
    rst = []
    for i in range(len(paths)):
        if i == constraint['agent']:
            continue
        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint['loc'][0] == curr:
                rst.append(i)
        else:  # edge constraint
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                    or constraint['loc'] == [curr, prev]:
                rst.append(i)
    return rst


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.ll_solver = a_star
        self.my_map = my_map
        # print("my_map:", my_map) # :) my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)
        # self.cost_map = cost_map

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []
        self.gen_node_list = []
        self.exp_node_list = []

        # compute heuristics for the low-level search
        self.heuristics = []

        compute_heuristic_start = time.time()
        for i in range(len(self.goals)):
            self.heuristics.append(compute_heuristics(my_map, self.goals[i]))
        # print("Compute heuristic spent: {}".format(time.time() - compute_heuristic_start))      

    def get_heuristics(self):
        return self.heuristics
    
    def get_closed_list(self):
        return self.gen_node_list, self.exp_node_list

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        self.push_node_close(node, "gen")
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1
    
    def push_node_close(self, node, flag):
        if flag == "gen":
            heapq.heappush(self.gen_node_list, (self.num_of_generated, node['cost'], len(node['collisions']), node))
        elif flag == "exp":
            heapq.heappush(self.exp_node_list, (self.num_of_expanded, node['cost'], len(node['collisions']), node))

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        self.push_node_close(node, "exp")
        # print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def make_root(self):
        AStar = A_Star
        
        """
        Generate the root node
        constraints   - list of constraints
        paths         - list of paths, one for each agent
                      [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        collisions     - list of collisions in paths
        """

        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}

        for i in range(self.num_of_agents):  # Find initial path for each agent
            astar = AStar(self.my_map, self.starts, self.goals, self.heuristics,i, root['constraints'])
            path = astar.find_paths()

            # path = ma_star(self.my_map, self.starts, self.goals, self.heuristics,[i], root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path[0])

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])

        return root

    def find_solution(self):
        """ 
        Finds paths for all agents from their start locations to their goal locations

        disjoint         - use disjoint splitting or not
        a_star_version   - version of A*; "a_star" or "pea_star"
        """

        self.start_time = time.time()
        astar_start = time.time()
        splitter = standard_splitting
        AStar = A_Star
        
        """
        Generate the root node
        constraints   - list of constraints
        paths         - list of paths, one for each agent
                      [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        collisions     - list of collisions in paths
        """

        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}

        for i in range(self.num_of_agents):  # Find initial path for each agent
            astar = AStar(self.my_map, self.starts, self.goals, self.heuristics,i, root['constraints'])
            path = astar.find_paths()

            # path = ma_star(self.my_map, self.starts, self.goals, self.heuristics,[i], root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path[0])

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        astar_end = time.time()
        print("CBS Astar spent: {}".format(astar_end - astar_start))

        """
        ##############################
        Task 3.3: High-Level Search

        Repeat the following as long as the open list is not empty:
        1. Get the next node from the open list (you can use self.pop_node()
        2. If this node has no collision, return solution
        3. Otherwise, choose the first collision and convert to a list of constraints (using your
            standard_splitting function). Add a new child node to your open list for each constraint

        Ensure to create a copy of any objects that your child nodes might inherit
        ###############################
        """

        high_level_start = time.time()
        while len(self.open_list) > 0:
            self.search_time = time.time()

            if self.search_time - self.start_time > 300:
                print("CBS execute too long")
                return None, None, None, None

            # Get A* solution
            p = self.pop_node()

            ### END POINT: return solution ###
            if p['collisions'] == []:
                # number of nodes generated/expanded for comparing implementations
                # print("CBS finished!!")
                high_level_end = time.time()
                high_level_CT = high_level_end - high_level_start
                total_CT = high_level_end - self.start_time
                print("CBS High level spent: {}".format(high_level_CT))
                return p['paths'], self.num_of_generated, self.num_of_expanded, total_CT

            # Get first collision
            collision = p['collisions'].pop(0)
            constraints = splitter(collision)
            # print("Constraints:", constraints)

            for constraint in constraints:
                # print("Constraint:",constraint)
                # Empty node q
                q = {'cost':0,
                    'constraints': [constraint],
                    'paths':[],
                    'collisions':[]
                    }
                
                # print(q)

                # Init q node
                for c in p['constraints']:
                    if c not in q['constraints']:
                        q['constraints'].append(c)
                
                for pa in p['paths']:
                    q['paths'].append(pa)
                

                ai = constraint['agent']
                astar = AStar(self.my_map, self.starts, self.goals, self.heuristics, ai, q['constraints'])
                path = astar.find_paths()

                if path is not None:
                    q['paths'][ai]= path[0]

                    # task 4
                    continue_flag = False
                    if constraint['positive']:
                        vol = paths_violate_constraint(constraint, q['paths'])
                        print("vol:", vol)

                        for v in vol:
                            astar_v = AStar(self.my_map,self.starts, self.goals,self.heuristics,v,q['constraints'])
                            path_v = astar_v.find_paths()
                            if path_v  is None:
                                continue_flag =True
                            else:
                                q['paths'][v] = path_v[0]
                        if continue_flag:
                            continue
                    q['collisions'] = detect_collisions(q['paths'])
                    q['cost'] = get_sum_of_cost(q['paths'])
                    self.push_node(q)
                
                # print("Node q:", q)
            
            # print("ONE ITER======================")

    def print_results(self, node):
        print("\n Found a solution! CBS~ \n")
        CPU_time = time.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Generated nodes: {}".format(self.num_of_generated))
        print("Expanded nodes:  {}".format(self.num_of_expanded))

        print("Solution:")
        for i in range(len(node['paths'])):
            print("agent", i, ": ", node['paths'][i])


############### WITH COST MAP !!! ########################


class CBSSolver_Cost(object):
    """The high-level search of CCBS."""

    def __init__(self, my_map, starts, goals, costmap_list):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.ll_solver = a_star
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)
        self.costmap_list = costmap_list

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []
        self.closed_list = []

        # compute heuristics for the low-level search
        self.heuristics = []

        # compute_heuristic_start = time.time()
        for i in range(len(self.goals)):
            self.heuristics.append(compute_heuristics_costmap(my_map, self.goals[i], costmap_list[i]))
        # print("Compute heuristic spent: {}".format(time.time() - compute_heuristic_start))

    def get_heuristics(self):
        return self.heuristics

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1
    
    def push_node_close(self, node):
        heapq.heappush(self.closed_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        # print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node


    def find_solution(self): #, cbs_gen_nodes, cbs_exp_nodes
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint         - use disjoint splitting or not
        a_star_version   - version of A*; "a_star" or "pea_star"
        """

        self.start_time = time.time()
        astar_start = time.time()
        splitter = standard_splitting
        AStar = A_Star

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths

        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        
        # print("Finding Astar Path")

        for i in range(self.num_of_agents):  # Find initial path for each agent
            astar = AStar(self.my_map, self.starts, self.goals, self.heuristics, i, root['constraints'])
            path = astar.find_paths()

            # path = ma_star(self.my_map, self.starts, self.goals, self.heuristics,[i], root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path[0])

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        astar_end = time.time()

        # print("Cost CBS Astar time spent: {}".format(astar_end - astar_start))

        # print("High-level start!")

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        high_level_start = time.time()
        while len(self.open_list) > 0:
            timer = time.time()
            p = self.pop_node()
            self.push_node_close(p)
            # print(self.closed_list)

            if timer - self.start_time > 180:
                return None,None,None,None
            
            if p['collisions'] == []:
                # self.print_results(p)
                high_level_end = time.time()
                high_level_CT = high_level_end - high_level_start
                total_CT = high_level_end - self.start_time
                # print("CBS High level spent: {}".format(high_level_CT))
                return p['paths'], self.num_of_generated, self.num_of_expanded, total_CT

            collision = p['collisions'].pop(0)
            constraints = splitter(collision)
            
            # print(constraints)

            for constraint in constraints:
                q = {'cost':0,
                    'constraints': [constraint],
                    'paths':[],
                    'collisions':[]
                }

                # print(q)

                for c in p['constraints']:
                    if c not in q['constraints']:
                        q['constraints'].append(c)

                for pa in p['paths']:
                    q['paths'].append(pa)
                
                ai = constraint['agent']
                astar = AStar(self.my_map, self.starts, self.goals, self.heuristics, ai, q['constraints'])
                path = astar.find_paths()

                if path is not None:
                    q['paths'][ai]= path[0]

                    # task 4
                    continue_flag = False
                    if constraint['positive']:
                        vol = paths_violate_constraint(constraint, q['paths'])

                        for v in vol:
                            astar_v = AStar(self.my_map, self.starts, self.goals, self.heuristics, v ,q['constraints'])
                            path_v = astar_v.find_paths()

                            if path_v  is None:
                                continue_flag =True

                        if continue_flag:
                            continue

                    q['collisions'] = detect_collisions(q['paths'])
                    q['cost'] = get_sum_of_cost(q['paths'])
                    self.push_node(q)


    def find_solution_heatmap(self, cbs_gen_nodes, cbs_exp_nodes): #
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint         - use disjoint splitting or not
        a_star_version   - version of A*; "a_star" or "pea_star"
        """

        self.start_time = time.time()
        astar_start = time.time()
        splitter = standard_splitting
        AStar = A_Star

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths

        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        
        # print("Finding Astar Path")

        for i in range(self.num_of_agents):  # Find initial path for each agent
            astar = AStar(self.my_map, self.starts, self.goals, self.heuristics, i, root['constraints'])
            path = astar.find_paths()

            # path = ma_star(self.my_map, self.starts, self.goals, self.heuristics,[i], root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path[0])

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        astar_end = time.time()

        print("Cost CBS Astar time spent: {}".format(astar_end - astar_start))

        # print("High-level start!")

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        high_level_start = time.time()
        while len(self.open_list) > 0:
            self.search_time = time.time()

            if self.search_time - self.start_time > 300:
                print("Cost CBS execute too long")
                return None, None, None, None
            
            if self.num_of_generated > cbs_gen_nodes or self.num_of_expanded > cbs_exp_nodes:
                print("Nodes increased!")
                return None, None, None, None
            
            p = self.pop_node()
            # print(p)
            
            if p['collisions'] == []:
                # self.print_results(p)
                high_level_end = time.time()
                high_level_CT = high_level_end - high_level_start
                total_CT = high_level_end - self.start_time
                print("CBS High level spent: {}".format(high_level_CT))
                return p['paths'], self.num_of_generated, self.num_of_expanded, total_CT

            collision = p['collisions'].pop(0)
            constraints = splitter(collision)
            
            for constraint in constraints:
                q = {'cost':0,
                    'constraints': [constraint],
                    'paths':[],
                    'collisions':[]
                }

                # print(q)

                for c in p['constraints']:
                    if c not in q['constraints']:
                        q['constraints'].append(c)

                for pa in p['paths']:
                    q['paths'].append(pa)
                
                ai = constraint['agent']
                astar = AStar(self.my_map, self.starts, self.goals, self.heuristics, ai, q['constraints'])
                path = astar.find_paths()

                if path is not None:
                    q['paths'][ai]= path[0]

                    # task 4
                    continue_flag = False
                    if constraint['positive']:
                        vol = paths_violate_constraint(constraint, q['paths'])

                        for v in vol:
                            astar_v = AStar(self.my_map, self.starts, self.goals, self.heuristics, v ,q['constraints'])
                            path_v = astar_v.find_paths()

                            if path_v  is None:
                                continue_flag =True

                        if continue_flag:
                            continue

                    q['collisions'] = detect_collisions(q['paths'])
                    q['cost'] = get_sum_of_cost(q['paths'])
                    self.push_node(q)

    def print_results(self, node):
        print("\n Found a solution! CCBS~ \n")
        CPU_time = time.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Generated nodes: {}".format(self.num_of_generated))
        print("Expanded nodes:  {}".format(self.num_of_expanded))

        print("Solution:")
        for i in range(len(node['paths'])):
            print("agent", i, ": ", node['paths'][i])

### For sorted heatmaps ###

class CBSSolver_Cost_Sorted(object):
    """The high-level search of CCBS."""

    def __init__(self, my_map, starts, goals, costmap_list):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.ll_solver = a_star
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)
        self.costmap_list = costmap_list

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []
        self.closed_list = []

        # compute heuristics for the low-level search
        self.heuristics = []

        # compute_heuristic_start = time.time()
        for i in range(len(self.goals)):
            self.heuristics.append(compute_heuristics_costmap(my_map, self.goals[i], costmap_list[i]))
        # print("Compute heuristic spent: {}".format(time.time() - compute_heuristic_start))

    def get_heuristics(self):
        return self.heuristics

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1
    
    def push_node_close(self, node):
        heapq.heappush(self.closed_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        # print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node


    def find_solution(self): #, cbs_gen_nodes, cbs_exp_nodes
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint         - use disjoint splitting or not
        a_star_version   - version of A*; "a_star" or "pea_star"
        """

        self.start_time = time.time()
        astar_start = time.time()
        splitter = standard_splitting
        AStar = A_Star

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths

        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        
        # print("Finding Astar Path")

        for i in range(self.num_of_agents):  # Find initial path for each agent
            astar = AStar(self.my_map, self.starts, self.goals, self.heuristics, i, root['constraints'])
            path = astar.find_paths()

            # path = ma_star(self.my_map, self.starts, self.goals, self.heuristics,[i], root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path[0])

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        astar_end = time.time()

        # print("Cost CBS Astar time spent: {}".format(astar_end - astar_start))

        # print("High-level start!")

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        high_level_start = time.time()
        while len(self.open_list) > 0:
            timer = time.time()
            p = self.pop_node()
            self.push_node_close(p)
            # print(self.closed_list)

            if timer - self.start_time > 60:
                return None,None,None,None
            
            if p['collisions'] == []:
                # self.print_results(p)
                high_level_end = time.time()
                high_level_CT = high_level_end - high_level_start
                total_CT = high_level_end - self.start_time
                # print("CBS High level spent: {}".format(high_level_CT))
                return p['paths'], self.num_of_generated, self.num_of_expanded, total_CT

            collision = p['collisions'].pop(0)
            constraints = splitter(collision)
            
            # print(constraints)

            for constraint in constraints:
                q = {'cost':0,
                    'constraints': [constraint],
                    'paths':[],
                    'collisions':[]
                }

                # print(q)

                for c in p['constraints']:
                    if c not in q['constraints']:
                        q['constraints'].append(c)

                for pa in p['paths']:
                    q['paths'].append(pa)
                
                ai = constraint['agent']
                astar = AStar(self.my_map, self.starts, self.goals, self.heuristics, ai, q['constraints'])
                path = astar.find_paths()

                if path is not None:
                    q['paths'][ai]= path[0]

                    # task 4
                    continue_flag = False
                    if constraint['positive']:
                        vol = paths_violate_constraint(constraint, q['paths'])

                        for v in vol:
                            astar_v = AStar(self.my_map, self.starts, self.goals, self.heuristics, v ,q['constraints'])
                            path_v = astar_v.find_paths()

                            if path_v  is None:
                                continue_flag =True

                        if continue_flag:
                            continue

                    q['collisions'] = detect_collisions(q['paths'])
                    q['cost'] = get_sum_of_cost(q['paths'])
                    self.push_node(q)


    def find_solution_heatmap(self, cbs_gen_nodes, cbs_exp_nodes): #
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint         - use disjoint splitting or not
        a_star_version   - version of A*; "a_star" or "pea_star"
        """

        self.start_time = time.time()
        astar_start = time.time()
        splitter = standard_splitting
        AStar = A_Star

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths

        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        
        # print("Finding Astar Path")

        for i in range(self.num_of_agents):  # Find initial path for each agent
            astar = AStar(self.my_map, self.starts, self.goals, self.heuristics, i, root['constraints'])
            path = astar.find_paths()

            # path = ma_star(self.my_map, self.starts, self.goals, self.heuristics,[i], root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path[0])

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        astar_end = time.time()

        print("Cost CBS Astar time spent: {}".format(astar_end - astar_start))

        # print("High-level start!")

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        high_level_start = time.time()
        while len(self.open_list) > 0:
            self.search_time = time.time()

            if self.search_time - self.start_time > 300:
                print("Cost CBS execute too long")
                return None, None, None, None
            
            if self.num_of_generated > cbs_gen_nodes or self.num_of_expanded > cbs_exp_nodes:
                print("Nodes increased!")
                return None, None, None, None
            
            p = self.pop_node()
            # print(p)
            
            if p['collisions'] == []:
                # self.print_results(p)
                high_level_end = time.time()
                high_level_CT = high_level_end - high_level_start
                total_CT = high_level_end - self.start_time
                print("CBS High level spent: {}".format(high_level_CT))
                return p['paths'], self.num_of_generated, self.num_of_expanded, total_CT

            collision = p['collisions'].pop(0)
            constraints = splitter(collision)
            
            for constraint in constraints:
                q = {'cost':0,
                    'constraints': [constraint],
                    'paths':[],
                    'collisions':[]
                }

                # print(q)

                for c in p['constraints']:
                    if c not in q['constraints']:
                        q['constraints'].append(c)

                for pa in p['paths']:
                    q['paths'].append(pa)
                
                ai = constraint['agent']
                astar = AStar(self.my_map, self.starts, self.goals, self.heuristics, ai, q['constraints'])
                path = astar.find_paths()

                if path is not None:
                    q['paths'][ai]= path[0]

                    # task 4
                    continue_flag = False
                    if constraint['positive']:
                        vol = paths_violate_constraint(constraint, q['paths'])

                        for v in vol:
                            astar_v = AStar(self.my_map, self.starts, self.goals, self.heuristics, v ,q['constraints'])
                            path_v = astar_v.find_paths()

                            if path_v  is None:
                                continue_flag =True

                        if continue_flag:
                            continue

                    q['collisions'] = detect_collisions(q['paths'])
                    q['cost'] = get_sum_of_cost(q['paths'])
                    self.push_node(q)

    def print_results(self, node):
        print("\n Found a solution! CCBS~ \n")
        CPU_time = time.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Generated nodes: {}".format(self.num_of_generated))
        print("Expanded nodes:  {}".format(self.num_of_expanded))

        print("Solution:")
        for i in range(len(node['paths'])):
            print("agent", i, ": ", node['paths'][i])

######## For Data plot! ###############################################################

class CBSSolver_Cost_Plot(object):
    """The high-level search of CCBS."""

    def __init__(self, my_map, starts, goals, costmap_list):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.ll_solver = a_star
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)
        self.costmap_list = costmap_list

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []
        self.exp_node_list = []
        self.gen_node_list = []

        # compute heuristics for the low-level search
        self.heuristics = []

        # compute_heuristic_start = time.time()
        for i in range(len(self.goals)):
            self.heuristics.append(compute_heuristics_costmap(my_map, self.goals[i], costmap_list[i]))
        # print("Compute heuristic spent: {}".format(time.time() - compute_heuristic_start))

    def get_heuristics(self):
        return self.heuristics
    
    def get_closed_list(self):
        return self.gen_node_list, self.exp_node_list

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        self.push_node_close(node, "gen")
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1
    
    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        self.push_node_close(node, "exp")
        # print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def push_node_close(self, node, flag):
        if flag == "gen":
            heapq.heappush(self.gen_node_list, (self.num_of_generated, node['cost'], len(node['collisions']), node))
        elif flag == "exp":
            heapq.heappush(self.exp_node_list, (self.num_of_expanded, node['cost'], len(node['collisions']), node))

    def find_solution(self): #
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint         - use disjoint splitting or not
        a_star_version   - version of A*; "a_star" or "pea_star"
        """
        print("Learning_CBS start")
        self.start_time = time.time()
        astar_start = time.time()
        splitter = standard_splitting
        AStar = A_Star

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths

        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        
        # print("Finding Astar Path")

        for i in range(self.num_of_agents):  # Find initial path for each agent
            astar = AStar(self.my_map, self.starts, self.goals, self.heuristics, i, root['constraints'])
            path = astar.find_paths()

            # path = ma_star(self.my_map, self.starts, self.goals, self.heuristics,[i], root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path[0])

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        astar_end = time.time()

        # print("Cost CBS Astar time spent: {}".format(astar_end - astar_start))

        # print("High-level start!")

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        high_level_start = time.time()
        while len(self.open_list) > 0:
            p = self.pop_node()



            
            if p['collisions'] == []:
                # self.print_results(p)
                high_level_end = time.time()
                high_level_CT = high_level_end - high_level_start
                total_CT = high_level_end - self.start_time
                # print("CBS High level spent: {}".format(high_level_CT))
                return p['paths'], self.num_of_generated, self.num_of_expanded, total_CT

            collision = p['collisions'].pop(0)
            constraints = splitter(collision)
            
            # print(constraints)

            for constraint in constraints:
                q = {'cost':0,
                    'constraints': [constraint],
                    'paths':[],
                    'collisions':[]
                }

                # print(q)

                for c in p['constraints']:
                    if c not in q['constraints']:
                        q['constraints'].append(c)

                for pa in p['paths']:
                    q['paths'].append(pa)
                
                ai = constraint['agent']
                astar = AStar(self.my_map, self.starts, self.goals, self.heuristics, ai, q['constraints'])
                path = astar.find_paths()

                if path is not None:
                    q['paths'][ai]= path[0]

                    # task 4
                    continue_flag = False
                    if constraint['positive']:
                        vol = paths_violate_constraint(constraint, q['paths'])

                        for v in vol:
                            astar_v = AStar(self.my_map, self.starts, self.goals, self.heuristics, v ,q['constraints'])
                            path_v = astar_v.find_paths()

                            if path_v  is None:
                                continue_flag =True

                        if continue_flag:
                            continue

                    q['collisions'] = detect_collisions(q['paths'])
                    q['cost'] = get_sum_of_cost(q['paths'])
                    self.push_node(q)
    
    def find_solution_gen_nodes(self, cbs_gen_nodes, cbs_exp_nodes): #
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint         - use disjoint splitting or not
        a_star_version   - version of A*; "a_star" or "pea_star"
        """

        self.start_time = time.time()
        astar_start = time.time()
        splitter = standard_splitting
        AStar = A_Star

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths

        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        
        # print("Finding Astar Path")

        for i in range(self.num_of_agents):  # Find initial path for each agent
            astar = AStar(self.my_map, self.starts, self.goals, self.heuristics, i, root['constraints'])
            path = astar.find_paths()

            # path = ma_star(self.my_map, self.starts, self.goals, self.heuristics,[i], root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path[0])

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        astar_end = time.time()

        # print("Cost CBS Astar time spent: {}".format(astar_end - astar_start))

        # print("High-level start!")

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        high_level_start = time.time()
        while len(self.open_list) > 0:
            p = self.pop_node()

            if self.num_of_generated > cbs_gen_nodes *2 :
                print("Too many nodes generated!!")
                return None, None, None, None
            
            if p['collisions'] == []:
                # self.print_results(p)
                high_level_end = time.time()
                high_level_CT = high_level_end - high_level_start
                total_CT = high_level_end - self.start_time
                # print("CBS High level spent: {}".format(high_level_CT))
                return p['paths'], self.num_of_generated, self.num_of_expanded, total_CT

            collision = p['collisions'].pop(0)
            constraints = splitter(collision)
            
            # print(constraints)

            for constraint in constraints:
                q = {'cost':0,
                    'constraints': [constraint],
                    'paths':[],
                    'collisions':[]
                }

                # print(q)

                for c in p['constraints']:
                    if c not in q['constraints']:
                        q['constraints'].append(c)

                for pa in p['paths']:
                    q['paths'].append(pa)
                
                ai = constraint['agent']
                astar = AStar(self.my_map, self.starts, self.goals, self.heuristics, ai, q['constraints'])
                path = astar.find_paths()

                if path is not None:
                    q['paths'][ai]= path[0]

                    # task 4
                    continue_flag = False
                    if constraint['positive']:
                        vol = paths_violate_constraint(constraint, q['paths'])

                        for v in vol:
                            astar_v = AStar(self.my_map, self.starts, self.goals, self.heuristics, v ,q['constraints'])
                            path_v = astar_v.find_paths()

                            if path_v  is None:
                                continue_flag =True

                        if continue_flag:
                            continue

                    q['collisions'] = detect_collisions(q['paths'])
                    q['cost'] = get_sum_of_cost(q['paths'])
                    self.push_node(q)


    def find_solution_heatmap(self, cbs_gen_nodes, cbs_exp_nodes): #
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint         - use disjoint splitting or not
        a_star_version   - version of A*; "a_star" or "pea_star"
        """

        self.start_time = time.time()
        astar_start = time.time()
        splitter = standard_splitting
        AStar = A_Star

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths

        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        
        # print("Finding Astar Path")

        for i in range(self.num_of_agents):  # Find initial path for each agent
            astar = AStar(self.my_map, self.starts, self.goals, self.heuristics, i, root['constraints'])
            path = astar.find_paths()

            # path = ma_star(self.my_map, self.starts, self.goals, self.heuristics,[i], root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path[0])

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        astar_end = time.time()

        print("Cost CBS Astar time spent: {}".format(astar_end - astar_start))

        # print("High-level start!")

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        high_level_start = time.time()
        while len(self.open_list) > 0:
            self.search_time = time.time()

            if self.search_time - self.start_time > 2100:
                print("Cost CBS execute too long")
                return None, None, None, None, None
            
            if self.num_of_generated > cbs_gen_nodes or self.num_of_expanded > cbs_exp_nodes:
                print("Nodes increased!")
                return None, None, None, None, None
            
            p = self.pop_node()
            # print(p)
            
            if p['collisions'] == []:
                # self.print_results(p)
                high_level_end = time.time()
                high_level_CT = high_level_end - high_level_start
                total_CT = high_level_end - self.start_time
                print("CBS High level spent: {}".format(high_level_CT))
                return p['paths'], self.num_of_generated, self.num_of_expanded, total_CT, high_level_CT

            collision = p['collisions'].pop(0)
            constraints = splitter(collision)
            
            for constraint in constraints:
                q = {'cost':0,
                    'constraints': [constraint],
                    'paths':[],
                    'collisions':[]
                }

                # print(q)

                for c in p['constraints']:
                    if c not in q['constraints']:
                        q['constraints'].append(c)

                for pa in p['paths']:
                    q['paths'].append(pa)
                
                ai = constraint['agent']
                astar = AStar(self.my_map, self.starts, self.goals, self.heuristics, ai, q['constraints'])
                path = astar.find_paths()

                if path is not None:
                    q['paths'][ai]= path[0]

                    # task 4
                    continue_flag = False
                    if constraint['positive']:
                        vol = paths_violate_constraint(constraint, q['paths'])

                        for v in vol:
                            astar_v = AStar(self.my_map, self.starts, self.goals, self.heuristics, v ,q['constraints'])
                            path_v = astar_v.find_paths()

                            if path_v  is None:
                                continue_flag =True

                        if continue_flag:
                            continue

                    q['collisions'] = detect_collisions(q['paths'])
                    q['cost'] = get_sum_of_cost(q['paths'])
                    self.push_node(q)

    def print_results(self, node):
        print("\n Found a solution! CCBS~ \n")
        CPU_time = time.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Generated nodes: {}".format(self.num_of_generated))
        print("Expanded nodes:  {}".format(self.num_of_expanded))

        print("Solution:")
        for i in range(len(node['paths'])):
            print("agent", i, ": ", node['paths'][i])


############### FOR DATA GENERATION !!! ########################

class CBSSolver_Cost_Data_generate(object):
    def __init__(self, my_map, starts, goals, costmap_list, nodes_gen_icbs, nodes_exp_icbs):
        self.ll_solver = a_star
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)
        self.costmap_list = costmap_list

        self.nodes_gen_icbs = nodes_gen_icbs
        self.nodes_exp_icbs = nodes_exp_icbs

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for i in range(len(self.goals)):
            self.heuristics.append(compute_heuristics_uniform_costmap(my_map, self.goals[i], costmap_list[i]))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        # print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node


    def find_solution(self, disjoint, a_star_version):
        self.start_time = time.time()
        
        if disjoint:
            splitter = disjoint_splitting
        else:
            splitter = standard_splitting

        print("USING: ", splitter)

        AStar = A_Star
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}

        for i in range(self.num_of_agents):  # Find initial path for each agent
            astar = AStar(self.my_map, self.starts, self.goals, self.heuristics,i, root['constraints'])
            path = astar.find_paths()

            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path[0])

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        while len(self.open_list) > 0:
            p = self.pop_node()

            if self.num_of_generated > self.nodes_gen_icbs or self.num_of_expanded > self.nodes_exp_icbs:
                return None, None, None
            
            if p['collisions'] == []:
                self.print_results(p)
                # for pa in p['paths']:
                #     print(pa)
                return p['paths'], self.num_of_generated, self.num_of_expanded # number of nodes generated/expanded for comparing implementations
            collision = p['collisions'].pop(0)
            
            constraints = splitter(collision)

            for constraint in constraints:
                q = {'cost':0,
                    'constraints': [constraint],
                    'paths':[],
                    'collisions':[]
                }
                for c in p['constraints']:
                    if c not in q['constraints']:
                        q['constraints'].append(c)
                for pa in p['paths']:
                    q['paths'].append(pa)
                
                ai = constraint['agent']
                astar = AStar(self.my_map,self.starts, self.goals,self.heuristics,ai,q['constraints'])
                path = astar.find_paths()

                if path is not None:
                    q['paths'][ai] = path[0]
                    # task 4
                    continue_flag = False
                    if constraint['positive']:
                        vol = paths_violate_constraint(constraint,q['paths'])
                        for v in vol:
                            astar_v = AStar(self.my_map,self.starts, self.goals,self.heuristics,v,q['constraints'])
                            path_v = astar_v.find_paths()
                            if path_v  is None:
                                continue_flag =True
                            else:
                                q['paths'][v] = path_v[0]
                        if continue_flag:
                            continue
                    q['collisions'] = detect_collisions(q['paths'])
                    q['cost'] = get_sum_of_cost(q['paths'])
                    self.push_node(q)     
        return None, None, None
        self.print_results(root)
        return root['paths']


    def print_results(self, node):
        print("\n Found a solution! CCBS~ \n")
        CPU_time = time.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Generated nodes: {}".format(self.num_of_generated))
        print("Expanded nodes:  {}".format(self.num_of_expanded))

        print("Solution:")
        for i in range(len(node['paths'])):
            print("agent", i, ": ", node['paths'][i])



class CBSSolver_best_costmap_generate(object):
    def __init__(self, my_map, starts, goals, costmap_list, nodes_gen_icbs, nodes_exp_icbs):
        self.ll_solver = a_star
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)
        self.costmap_list = costmap_list

        self.nodes_gen_icbs = nodes_gen_icbs
        self.nodes_exp_icbs = nodes_exp_icbs

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for i in range(len(self.goals)):
            self.heuristics.append(compute_heuristics_uniform_costmap(my_map, self.goals[i], costmap_list[i]))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node


    def find_solution(self, disjoint, a_star_version):
        self.start_time = time.time()
        
        if disjoint:
            splitter = disjoint_splitting
        else:
            splitter = standard_splitting

        print("USING: ", splitter)

        AStar = A_Star
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}

        for i in range(self.num_of_agents):  # Find initial path for each agent
            astar = AStar(self.my_map, self.starts, self.goals, self.heuristics,i, root['constraints'])
            path = astar.find_paths()

            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path[0])

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        while len(self.open_list) > 0:
            p = self.pop_node()

            if self.num_of_generated > self.nodes_gen_icbs or self.num_of_expanded > self.nodes_exp_icbs:
                return None, None, None
            
            if p['collisions'] == []:
                self.print_results(p)
                # for pa in p['paths']:
                #     print(pa)
                return p['paths'], self.num_of_generated, self.num_of_expanded # number of nodes generated/expanded for comparing implementations
            collision = p['collisions'].pop(0)
            
            constraints = splitter(collision)

            for constraint in constraints:
                q = {'cost':0,
                    'constraints': [constraint],
                    'paths':[],
                    'collisions':[]
                }
                for c in p['constraints']:
                    if c not in q['constraints']:
                        q['constraints'].append(c)
                for pa in p['paths']:
                    q['paths'].append(pa)
                
                ai = constraint['agent']
                astar = AStar(self.my_map,self.starts, self.goals,self.heuristics,ai,q['constraints'])
                path = astar.find_paths()

                if path is not None:
                    q['paths'][ai]= path[0]
                    # task 4
                    continue_flag = False
                    if constraint['positive']:
                        vol = paths_violate_constraint(constraint,q['paths'])
                        for v in vol:
                            astar_v = AStar(self.my_map,self.starts, self.goals,self.heuristics,v,q['constraints'])
                            path_v = astar_v.find_paths()
                            if path_v  is None:
                                continue_flag =True
                            else:
                                q['paths'][v] = path_v[0]
                        if continue_flag:
                            continue
                    q['collisions'] = detect_collisions(q['paths'])
                    q['cost'] = get_sum_of_cost(q['paths'])
                    self.push_node(q)     
        return None, None, None
        self.print_results(root)
        return root['paths']


    def print_results(self, node):
        print("\n Found a solution! CCBS~ \n")
        CPU_time = time.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Generated nodes: {}".format(self.num_of_generated))
        print("Expanded nodes:  {}".format(self.num_of_expanded))

        print("Solution:")
        for i in range(len(node['paths'])):
            print("agent", i, ": ", node['paths'][i])