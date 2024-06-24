import pandas as pd
import numpy as np
import networkx as nx
import random

def graph_to_egonet_features(
    G,
    FULL_egonets=True,
    IN_egonets=False,
    summable_attributes=[],
    verbose=False,
):
    """Extract egonet features from input graph.
    
    Parameters
    ----------
    G : networkx DiGraph or pandas DataFrame
        Transactions graph. It must have the following edge attributes:
        * 'cum_amount': the cumulated amount of transactions from source to destination node
        * 'nr_transactions': number of transactions between the two nodes
        It can have other attributes.
        A dataframe is immediately converted to Networkx graph, so there is no advantage in giving a dataframe.
        The columns for nodes forming an edge should be named "id_source" and "id_destination".
        
    FULL_egonets : bool, default=True
        Whether full undirected egonets (with radius 1) are considered
    
    IN_egonets : bool, default=False
        Whether in or out egonets are considered, with respect to the current node.
        Is considered only if FULL_egonets=False.

    summable_attributes: list
        List of edge attributes to be summed as adjacent nodes features.
        The name of the feature is the same as the name of the edge attribute.
        The attributes must be present in all edges; no check is made.
        
    verbose : bool, default=False
        To control the verbosity of the procedure.

    Returns
    -------
    node_features : pandas dataframe
        Contains the following columns.
        * 'id': node id from G
        * 'f_degree_in': in degree of the node
        * 'f_degree_out': out degree of the node
        * 'f_amount_in': total amount received by node from neighbors
        * 'f_amount_out': total amount sent by node to neighbors
        * 'f_nr_trans_in'
        * 'f_nr_trans_out'
        * 'f_ego_nr_nodes'
        * 'f_ego_nr_edges',
        * 'f_egored_degree_in'
        * 'f_egored_degree_out'
        * 'f_egored_amount_in'
        * 'f_egored_amount_out'
        * 'f_egored_nr_trans_in'
        * 'f_egored_nr_trans_out
        * 'f_egored_nr_nodes'
        * 'f_egored_nr_edges'
        * other columns indicated by summable attributes
    """

    # BD 19.08.2021

    if isinstance(G, pd.DataFrame):     # if dataframe, convert to graph
        G = nx.from_pandas_edgelist(df=G, source='id_source', target='id_destination',
                            edge_attr=True, create_using=nx.DiGraph)
    
    if not FULL_egonets and IN_egonets:  # reverse graph if in egonets are desired
      G = G.reverse(copy = False)

    if verbose:
        info = nx.info(G)
        print("Graph info", info)

    # Build matrix of node features with the following columns (this is faster than using a dataframe)
    #  0 - node id
    #  1 - degree in
    #  2 - degree out
    #  3 - amount in
    #  4 - amount out
    #  5 - number of transactions to node
    #  6 - number of transactions from node
    #  7 - egonet number of nodes
    #  8 - egonet nubber of edges
    #  9-16 - same as 1-8, but for reduced egonets (after removing lonely edges)
    #  17-  - use specified summable attributes 

    feat_id = 0
    feat_ego_nr_nodes = 7
    feat_ego_nr_edges = 8
    feat_egored_nr_nodes = 15
    feat_egored_nr_edges = 16
    if not FULL_egonets and IN_egonets: # in egonets, only where edges are reversal matters
        feat_degree_in = 2
        feat_degree_out = 1
        feat_amount_in = 4
        feat_amount_out = 3
        feat_trans_nr_in = 6
        feat_trans_nr_out = 5
        feat_egored_degree_in = 10
        feat_egored_degree_out = 9
        feat_egored_amount_in = 12
        feat_egored_amount_out = 11
        feat_egored_trans_nr_in = 14
        feat_egored_trans_nr_out = 13
    else:                               # full or out egonets
        feat_degree_in = 1
        feat_degree_out = 2
        feat_amount_in = 3
        feat_amount_out = 4
        feat_trans_nr_in = 5
        feat_trans_nr_out = 6
        feat_egored_degree_in = 9
        feat_egored_degree_out = 10
        feat_egored_amount_in = 11
        feat_egored_amount_out = 12
        feat_egored_trans_nr_in = 13
        feat_egored_trans_nr_out = 14

    Nn = G.number_of_nodes()
    Ns = len(summable_attributes)
    Nef = 17            # number of egonets features (16) and id
    Nf = Nef + Ns       # total number of features: egonets + number of user specified attributes

    node_features = np.zeros((Nn, Nf))  # store features in matrix (it's faster)

    # go over all nodes and extract features
    row = 0
    for node in G:
        if FULL_egonets:    # generate full egonet with ad hoc construction (networkx is slow!)
            No = G.successors(node)
            Ni = G.predecessors(node)
            enodes = [node] # create list of egonets nodes
            enodes.extend(No)
            enodes.extend(Ni)
            Gs = G.subgraph(enodes)
        else:  # out or in egonets
            Gs = nx.generators.ego.ego_graph(G, node, radius=1)

        node_features[row, feat_id] = node
        # in and out degree (that's easy) and total number of edges
        node_features[row, feat_degree_in] = Gs.in_degree(node)
        node_features[row, feat_degree_out] = Gs.out_degree(node)
        node_features[row, feat_ego_nr_nodes] = len(Gs)
        node_features[row, feat_ego_nr_edges] = Gs.to_undirected().number_of_edges()

        # Amounts in and out. Also, sums of user specified attributes
        amount_out = 0
        amount_in = 0
        trans_in = 0
        trans_out = 0
        s = np.zeros(Ns);
        for e in Gs.edges():   # go over edges and simply add amounts, trans.nr., user spec. attr.
            edge_data = Gs.get_edge_data(e[0], e[1])
            if e[0] == node:     # our node can be source
                amount_out += edge_data["cum_amount"]
                trans_out += edge_data["nr_transactions"]
                for i in range(Ns):
                  s[i] += edge_data[summable_attributes[i]]
            if e[1] == node:     # or destination
                amount_in = amount_in + edge_data["cum_amount"]
                trans_in = trans_in + edge_data["nr_transactions"]
                for i in range(Ns):
                  s[i] += edge_data[summable_attributes[i]]
        node_features[row, feat_amount_in] = amount_in
        node_features[row, feat_amount_out] = amount_out
        node_features[row, feat_trans_nr_in] = trans_in
        node_features[row, feat_trans_nr_out] = trans_out
        for i in range(Ns):
            node_features[row, Nef+i] = s[i]

        # Same operations on reduced egonets, which are obtained by removing lonely nodes from egonet
        # (a lonely node is connected only with the central node of the egonet)
        enodes = np.zeros(len(Gs))
        i_ego = 0
        for nn in Gs:
            if Gs.degree(nn) > 1: # keep only nodes that have more than one neighbor
                enodes[i_ego] = nn
                i_ego += 1
        if i_ego > 0:                         # if something left in the egonet (i.e., not a star)
            Gs = Gs.subgraph(enodes[0:i_ego]) # the reduced egonet
        
            # Repeat the same operations as for the full egonet
            # In and out degree, total number of edges (they are smaller now, of course)
            node_features[row, feat_egored_degree_in] = Gs.in_degree(node)
            node_features[row, feat_egored_degree_out] = Gs.out_degree(node)
            node_features[row, feat_egored_nr_nodes] = len(Gs)
            node_features[row, feat_egored_nr_edges] = Gs.to_undirected().number_of_edges()

            # Amounts in and out (they are also smaller and should be much smaller normally)
            amount_out = 0
            amount_in = 0
            trans_in = 0
            trans_out = 0
            for e in Gs.edges():   # go over edges and simply add amounts
                edge_data = Gs.get_edge_data(e[0], e[1])
                if e[0] == node:     # our node can be source
                    amount_out = amount_out + edge_data["cum_amount"]
                    trans_out = trans_out + edge_data["nr_transactions"]
                if e[1] == node:     # or destination
                    amount_in = amount_in + edge_data["cum_amount"]
                    trans_in = trans_in + edge_data["nr_transactions"]
            node_features[row, feat_egored_amount_in] = amount_in
            node_features[row, feat_egored_amount_out] = amount_out
            node_features[row, feat_egored_trans_nr_in] = trans_in
            node_features[row, feat_egored_trans_nr_out] = trans_out

        # that's all, go to next node
        row = row + 1
        if verbose:
            if row%1000 == 0 or row == Nn:
                print("\r", "Nodes processed: ", row, end='\r', sep='', flush=True)

    if verbose:
        print('\r')

    # sort on id
    node_features = node_features[np.argsort(node_features[:,0],0)]

    if not FULL_egonets and IN_egonets:
        G = G.reverse(copy = False)     # reverse back the graph, just in case
    
    # convert matrix to dataframe
    df_columns = ["id", "f_degree_in", "f_degree_out", "f_amount_in", "f_amount_out",
                  "f_nr_trans_in", "f_nr_trans_out", "f_ego_nr_nodes", "f_ego_nr_edges",
                  "f_egored_degree_in", "f_egored_degree_out",
                  "f_egored_amount_in", "f_egored_amount_out",
                  "f_egored_nr_trans_in", "f_egored_nr_trans_out",
                  "f_egored_nr_nodes", "f_egored_nr_edges"] + summable_attributes
    node_features = pd.DataFrame(node_features, columns=df_columns)

    return node_features


#---------------------------------------------------------------------------------
# 
# Input:  dir      - 0 - out (from the node), 1 - in (to the node)
def single_rwalk(
    G,
    node,
    rwalk_len,
    root_desc,
    out_walk = True,
    prob_edge = None,
):
    """Single random walk results.

    Parameters
    ----------
    G : networkx DiGraph
        Transactions graph. It must have the following edge attribute:
        * 'cum_amount': the cumulated amount of transactions from source to destination node
   
    node : int
        Start node
    
    rwalk_len : int
        The maximum length of the random walk
    
    root_desc : vector of neighbor nodes
        If empty, it consists of all neighbors (successors or predecessors, depending on out_walk)
        
    out_walk : bool
        Direction of the random walk (True - from the node, False - to the node)

    prob_edge : string, default=None
        How to randomly choose the next node:
        * None: equal probabilities for all neighbors
        * 'cum_amount': probability is proportional with 'cum_amount'
        * 'average_amount': probability is proportional with average amount (not yet implemented!!!)

    Returns
    -------
    amount_start : real
        Amount on the first leg of the walk
        
    amount_thru_walk : real
        Amount that goes through the whole walk, from start to finish (the minimum on all legs)
        
    amount_back : real
        Amount that comes back to the start node

    """
                
    # BD 20.08.2021
    #    18.12.2021: added probability proportional with sum
    
    if len(root_desc)==0:     # initialize neighbor list
        if out_walk:
            root_desc = np.asarray(list(G.successors(node)))
        else:
            root_desc = np.asarray(list(G.predecessors(node)))
            
    amount_thru_walk = np.inf  # minimum amount on the random walk (flow on that path)
    amount_back = 0      # amount that comes back to starting node
    desc = root_desc
    ndesc = root_desc.size
    crt_node = node
    for iw in range(rwalk_len):
        # add probabilities to edges if so required
        if prob_edge == None:   # equal probability between neighbors
            next_node = desc[random.randrange(ndesc)]
        else:  # for the moment we implement directly 'cum_amount'
            n_prob = np.zeros(ndesc)
            for ii in range(ndesc):
                if out_walk:
                    n_prob[ii] = G.get_edge_data(crt_node, desc[ii])['cum_amount']
                else:
                    n_prob[ii] = G.get_edge_data(desc[ii], crt_node)['cum_amount']
            n_prob = n_prob / np.sum(n_prob)  # normalize probabilities to 1
            next_node = np.random.choice(desc, p=n_prob)
            
        if out_walk:
            a = G.get_edge_data(crt_node, next_node)['cum_amount']
        else:
            a = G.get_edge_data(next_node, crt_node)['cum_amount']
        if iw == 0:    # amount on first hop of the random walk
            amount_start = a
        if a < amount_thru_walk:
            amount_thru_walk = a # amount that goes all the way to the end (it's the minimum)
        if next_node == node: # we are back in the starting node, money are back
            amount_back = amount_thru_walk
            break
        if iw < rwalk_len-1:  # if not the last edge of the walk, prepare for next step
            crt_node = next_node
            if out_walk:
                desc = np.asarray(list(G.successors(crt_node)))
            else:
                desc = np.asarray(list(G.predecessors(crt_node)))
        ndesc = desc.size
        if ndesc == 0: # we are in a sink
            break
    return amount_start, amount_thru_walk, amount_back

#---------------------------------------------------------------------------------
def graph_to_rwalk_features(
    G,
    rwalk_len,
    rwalk_reps,
    prob_edge = None,
    verbose = False,
):
    """Extract random walk features from input graph.

     Parameters
    ----------
    G : networkx DiGraph or pandas DataFrame
        Transactions graph. It must have the following edge attributes:
        * 'cum_amount': the cumulated amount of transactions from source to destination node
        * 'nr_transactions': number of transactions between the two nodes
        It can have other attributes.
        A dataframe is immediately converted to Networkx graph, so there is no advantage in giving a dataframe.
        The columns for nodes forming an edge should be named "id_source" and "id_destination".
    
    rwalk_len : int
        Length of the random walk.
    
    rwalk_reps : int
        Number of random walks starting from the same node.
    
    prob_edge : string, default=None
        How to randomly choose the next node:
        * None: equal probabilities for all neighbors
        * 'cum_amount': probability is proportional with 'cum_amount'
        * 'average_amount': probability is proportional with average amount (not yet implemented!!!)
                    
    verbose : bool, default=False
        To control the verbosity of the procedure.

    Returns
    -------
    node_features : pandas dataframe
        Contains the following columns:
        * 'id': node id from G
        * 'f_rwalk_start_amount': average amount on the first leg of the random walk (starting from the node)
        * 'f_rwalk_transfer_out': average amount going through the whole walk (the minimum on all legs)
        * 'f_rwalk_out_back': average amount coming back to the node
        * 'f_rwalk_out_back_max': maximum amount coming back to the node ("best ring")
        * 'f_rwalk_end_amount': same as above, but for random walks finishing in the node
        * 'f_rwalk_transfer_in'
        * 'f_rwalk_in_back'
        * 'f_rwalk_in_back_max'

    """
    # BD 20.08.2021
    #    18.12.2021: added probability proportional with sum

    if isinstance(G, pd.DataFrame):     # if dataframe, convert to graph
        G = nx.from_pandas_edgelist(
                df=G,
                source='id_source',
                target='id_destination',
                edge_attr=True,
                create_using=nx.DiGraph)
            
    N = G.number_of_nodes()
    node_features = np.zeros((N,9))  # matrix for storing random walk features (first column is node number)
    inode = 0
    for node in G:
        node_features[inode,0] = node
        root_desc = np.asarray(list(G.successors(node)))  # vector of neighbors
        if root_desc.size > 0:  # if the node has no successors, there is nothing to do
            rwalk_start_amounts = np.zeros(rwalk_reps)
            rwalk_thru_amounts = np.zeros(rwalk_reps)
            rwalk_back_amounts = np.zeros(rwalk_reps)

            # generate the random walks - starting from the node
            for ir in range(rwalk_reps):
                rwalk_start_amounts[ir], rwalk_thru_amounts[ir], rwalk_back_amounts[ir] = \
                        single_rwalk(G, node, rwalk_len, root_desc, True, prob_edge)
        
            node_features[inode,1] = np.average(rwalk_start_amounts)
            node_features[inode,2] = np.average(rwalk_thru_amounts)
            node_features[inode,3] = np.average(rwalk_back_amounts)
            node_features[inode,4] = np.max(rwalk_back_amounts)

        # same operations, backward from the starting node (==> walks finishing in the node)
        root_desc = np.asarray(list(G.predecessors(node)))  # we keep the same variables, but the meaning is different
        if root_desc.size > 0:  # if the node has no predecessors, there is nothing to do
            rwalk_start_amounts = np.zeros(rwalk_reps)
            rwalk_thru_amounts = np.zeros(rwalk_reps)
            rwalk_back_amounts = np.zeros(rwalk_reps)

            for ir in range(rwalk_reps):
                rwalk_start_amounts[ir], rwalk_thru_amounts[ir], rwalk_back_amounts[ir] = \
                        single_rwalk(G, node, rwalk_len, root_desc, False, prob_edge)

            node_features[inode,5] = np.average(rwalk_start_amounts)
            node_features[inode,6] = np.average(rwalk_thru_amounts)
            node_features[inode,7] = np.average(rwalk_back_amounts)
            node_features[inode,8] = np.max(rwalk_back_amounts)

        inode += 1
        if verbose:
            if inode % 1000 == 0 or inode == N:
                print("\r", "Nodes processed: ", inode, end='\r', sep='', flush=True)

    if verbose:
        print('\r')

    # sort on id
    node_features = node_features[np.argsort(node_features[:,0],0)]

    # convert matrix to dataframe
    df_columns = [
        "id",
        "f_rwalk_start_amount",
        "f_rwalk_transfer_out",
        "f_rwalk_out_back",
        "f_rwalk_out_back_max",
        "f_rwalk_end_amount",
        "f_rwalk_transfer_in",
        "f_rwalk_in_back",
        "f_rwalk_in_back_max"]
    node_features = pd.DataFrame(node_features, columns=df_columns)

    return node_features

#---------------------------------------------------------------------------------
def add_extra_egonet_features(
    egonet_features,
    feature_list,
    node_features = None,
):
    """Build node feature dataframe given the list of features
 
    Parameters
    ----------
    egonet_features : pandas dataframe
        Basic egonet features computed for a graph.
        May contain other features.
    
    feature_list : list of strings
        List of features, each given by its Graphomaly name.
        See the full list here???
    
    node_features : dataframe
        If given, the new feature values are appended to node_features.
        Otherwise, a new dataframe is created.
        
    Returns
    -------
    node_features : pandas dataframe
        Values of the new features
    """
    # BD 28.08.2021
    
    # set dataframe to store feature values as new columns
    if node_features is None:
        node_features = pd.DataFrame()     # create new dataframe and copy id column
        node_features["id"] = egonet_features["id"].copy()

    feat_found_list = []
    for feat in feature_list:  # take features one by one and compute if name is recognized
        feature_found = 1
        if feat == "f_average_amount_in":
            node_features[feat] = np.zeros((egonet_features.shape[0],))    # init with zeros
            ii = egonet_features["f_nr_trans_in"] > 0           # indices where computation is possible
            node_features.loc[ii, feat] = egonet_features.loc[ii, "f_amount_in"] / \
                egonet_features.loc[ii, "f_nr_trans_in"]        # the new feature
        elif feat == "f_average_amount_out":
            node_features[feat] = np.zeros((egonet_features.shape[0],)) 
            ii = egonet_features["f_nr_trans_out"] > 0
            node_features.loc[ii, feat] = egonet_features.loc[ii, "f_amount_out"] / \
                egonet_features.loc[ii, "f_nr_trans_out"]
        elif feat == "f_egored_average_amount_in":
            node_features[feat] = np.zeros((egonet_features.shape[0],)) 
            ii = egonet_features["f_egored_nr_trans_in"] > 0
            node_features.loc[ii, feat] = egonet_features.loc[ii, "f_egored_amount_in"] / \
                egonet_features.loc[ii, "f_egored_nr_trans_in"]
        elif feat == "f_egored_average_amount_out":
            node_features[feat] = np.zeros((egonet_features.shape[0],)) 
            ii = egonet_features["f_egored_nr_trans_out"] > 0
            node_features.loc[ii, feat] = egonet_features.loc[ii, "f_egored_amount_out"] / \
                egonet_features.loc[ii, "f_egored_nr_trans_out"]
        elif feat == "f_egored_degree_in_rel":
            node_features[feat] = np.zeros((egonet_features.shape[0],)) 
            ii = egonet_features["f_degree_in"] > 0
            node_features.loc[ii, feat] = egonet_features.loc[ii, "f_egored_degree_in"] / \
                egonet_features.loc[ii, "f_degree_in"]
        elif feat == "f_egored_degree_out_rel":
            node_features[feat] = np.zeros((egonet_features.shape[0],)) 
            ii = egonet_features["f_degree_out"] > 0
            node_features.loc[ii, feat] = egonet_features.loc[ii, "f_egored_degree_out"] / \
                egonet_features.loc[ii, "f_degree_out"]
        elif feat == "f_egored_amount_in_rel":
            node_features[feat] = np.zeros((egonet_features.shape[0],)) 
            ii = egonet_features["f_amount_in"] > 0
            node_features.loc[ii, feat] = egonet_features.loc[ii, "f_egored_amount_in"] / \
                egonet_features.loc[ii, "f_amount_in"]
        elif feat == "f_egored_amount_out_rel":
            node_features[feat] = np.zeros((egonet_features.shape[0],)) 
            ii = egonet_features["f_amount_out"] > 0
            node_features.loc[ii, feat] = egonet_features.loc[ii, "f_egored_amount_out"] / \
                egonet_features.loc[ii, "f_amount_out"]
        elif feat == "f_egored_average_amount_in_rel":
            node_features[feat] = np.zeros((egonet_features.shape[0],)) 
            ii = egonet_features["f_egored_nr_trans_in"] > 0
            node_features.loc[ii, feat] = egonet_features.loc[ii,"f_egored_amount_in"] / egonet_features.loc[ii,"f_egored_nr_trans_in"] / \
                (egonet_features.loc[ii, "f_amount_in"] / egonet_features.loc[ii, "f_nr_trans_in"])
        elif feat == "f_egored_average_amount_out_rel":  
            node_features[feat] = np.zeros((egonet_features.shape[0],)) 
            ii = egonet_features["f_egored_nr_trans_out"] > 0
            node_features.loc[ii, feat] = egonet_features.loc[ii,"f_egored_amount_out"] / egonet_features.loc[ii,"f_egored_nr_trans_out"] / \
                (egonet_features.loc[ii,"f_amount_out"] / egonet_features.loc[ii,"f_nr_trans_out"])
        elif feat == "f_egored_nr_edges_rel":
            node_features[feat] = np.zeros((egonet_features.shape[0],)) 
            ii = egonet_features["f_ego_nr_edges"] > 0
            node_features.loc[ii, feat] = egonet_features.loc[ii, "f_egored_nr_edges"] / \
                egonet_features.loc[ii, "f_ego_nr_edges"]
        elif feat == "f_egored_nr_nodes_rel":
            node_features[feat] = np.zeros((egonet_features.shape[0],)) 
            ii = egonet_features["f_ego_nr_nodes"] > 0
            node_features.loc[ii, feat] = egonet_features.loc[ii, "f_egored_nr_nodes"] / \
                egonet_features.loc[ii, "f_ego_nr_nodes"]
        elif feat == "f_ego_edge_density":
            node_features[feat] = egonet_features["f_ego_nr_edges"] / egonet_features["f_ego_nr_nodes"]
        elif feat == "f_egored_edge_density":
            node_features[feat] = np.zeros((egonet_features.shape[0],)) 
            ii = egonet_features["f_egored_nr_nodes"] > 0
            node_features.loc[ii, feat] = egonet_features.loc[ii, "f_egored_nr_edges"] / \
                egonet_features["f_egored_nr_nodes"]
        elif feat == "f_log2_amount_in":                    
            node_features[feat] = np.log2(egonet_features["f_amount_in"]+1)
        elif feat == "f_log2_amount_out":
            node_features[feat] = np.log2(egonet_features["f_amount_out"]+1)
        else:   # feature name is not recognized: ignore and just print a message, later
            feature_found = 0
            #print("Feature", feat, "not found")

        if feature_found:
            feat_found_list.append(feat)

    for feat in feat_found_list:
        feature_list.remove(feat)
        
    return node_features


#---------------------------------------------------------------------------------
def add_extra_rwalk_features(
    rwalk_features,
    feature_list,
    node_features = None,
):
    """Build node feature dataframe given the list of features
 
    Parameters
    ----------
    rwalk_features : pandas dataframe
        Some random walk features computed for a graph.
        May contain other features.
    
    feature_list : list of strings
        List of features, each given by its Graphomaly name.
        See the full list here???
    
    node_features : dataframe
        If given, the new feature values are appended to node_features.
        Otherwise, a new dataframe is created.
        
    Returns
    -------
    node_features : pandas dataframe
        Values of the new features
    """
    # BD 26.09.2021
    
    # set dataframe to store feature values as new columns
    if node_features is None:
        node_features = pd.DataFrame()     # create new dataframe and copy id column
        node_features["id"] = rwalk_features["id"].copy()

    feat_found_list = []
    for feat in feature_list:  # take features one by one and compute if name is recognized
        feature_found = 1
        if feat == "f_rwalk_ring_max":
            node_features[feat] = np.maximum(
                rwalk_features["f_rwalk_out_back_max"],
                rwalk_features["f_rwalk_in_back_max"])
        elif feat == "f_rwalk_ring_average":
            node_features[feat] = (rwalk_features["f_rwalk_out_back"] + rwalk_features["f_rwalk_in_back"])/2
        else:   # feature name is not recognized: ignore and just print a message
            feature_found = 0
            #print("Feature", feat, "not found")

        if feature_found:
            feat_found_list.append(feat)

    for feat in feat_found_list:
        feature_list.remove(feat)

    return node_features


#---------------------------------------------------------------------------------
def build_node_features_df(
    feature_list,
    df_list,
):
    """Build node feature dataframe given the list of features
 
    Parameters
    ----------
    feature_list : list of strings
        List of features, each given by is Graphomaly name.
        See the full list here???
    
    df_list : list of dataframes
        Precomputed feature values, stored in one or several dataframes.
        If more dataframes are used, the order of the nodes must be the same.
        At least one of the dataframes must have a column called "id".
        
    Returns
    -------
    node_features : pandas dataframe
        Values of the features
    """
    # BD 28.08.2021

    # find if id column exists in a dataframe and copy it
    id_col_found = False
    for df in df_list:
        df_col = df.columns
        if "id" in set(df_col):
            node_features = pd.DataFrame()
            node_features["id"] = df["id"].copy()
            id_col_found = True
            break
    
    if not id_col_found:
        return None                 # what's the best way to return errors ???
    
    f_list = feature_list.copy()   # working copy of the feature list
    # copy from the dataframes whatever feature can be found
    for feat in feature_list:
        for df in df_list:
            df_col = df.columns
            #print(df_col)
            if feat in set(df_col): # the feature is in that dataframe, as it is
                node_features[feat] = df[feat].copy()   # copy the column
                f_list.remove(feat)                     # and trim the list
                #print(f_list)

    # all remaining features are derivates of the basic features (now only egonets features)
    for df in df_list:
        df_col = df.columns
        if "f_ego_nr_nodes" in set(df_col):         # check if it's the egonet features dataframe
            node_features = add_extra_egonet_features(df, f_list, node_features)   # if so, compute remaining features
        if df_col[1].find("rwalk_") >= 0:            # check if it's a random walk feature
            node_features = add_extra_rwalk_features(df, f_list, node_features)
            
    if len(f_list) > 0:
        print("Features not found:", f_list)
            
    return node_features