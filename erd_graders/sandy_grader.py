import spacy
import textdistance
import networkx as nx

def compare_similarity(g,g_sol):
    nlp = spacy.load("en_core_web_lg")
    total_entity_sol = 0
    total_entity_submission = 0
    total = 0
    for i in range(g.number_of_nodes()):
        if g.nodes[i]['shape'] == "s":
            total_entity_submission +=1
            #node is an entity
            entity_name = g.nodes[i]['ocr']
            print("entity_name: "+entity_name)
            token = nlp(g.nodes[i]['ocr'])
            sol_entity_name = ""
            sol_entity_num = -1
            submission_num = -1
            for j in range(g_sol.number_of_nodes()):
                if g_sol.nodes[i]['shape'] == "s":
                    if(token.similarity(nlp(g_sol.nodes[i]['ocr']))>0.9 or textdistance.levenshtein.normalized_similarity(g_sol.nodes[j]['ocr'],entity_name)>0.85):
                        total_entity_sol += 1
                        #high similarity entity (similiar words or similar string)
                        print("Entity Name:"+g_sol.nodes[j]['ocr'])
                        sol_entity_name = g_sol.nodes[j]['ocr']
                        sol_entity_num = j
                        submission_num = i
                        break
            if (sol_entity_num!=-1):
                #found similar entity in solution graph
                sol_connected = list(nx.node_connected_component(g_sol, sol_entity_num))
                submission_connected = list(nx.node_connected_component(g, submission_num))
                solution_long = False
                mark = 0
                for i in submission_connected:
                    if(g.nodes[i]['shape']=='o'):
                        #The node is an attribute
                        similarity = 0
                        token = nlp(g.nodes[i]['ocr'])
                        print("token: "+g.nodes[i]['ocr'])
                        for j in sol_connected:
                            if(g_sol.nodes[j]['shape']=='o'):
                                #print("distance: "+str(textdistance.levenshtein.normalized_similarity(g.nodes[i]['ocr'],g_sol.nodes[j]['ocr'])))
                                #print("similarity: ",token.similarity(nlp(g_sol.nodes[j]['ocr'])))
                                ans_similarity = max(token.similarity(nlp(g_sol.nodes[j]['ocr'])),textdistance.levenshtein.normalized_similarity(g.nodes[i]['ocr'],g_sol.nodes[j]['ocr'])) # Get the max of the similarity mark
                                if(ans_similarity>similarity):
                                    similarity = ans_similarity
                        print("attribute mark earn: ",mark)
                        mark += similarity
                mark = mark / len(sol_connected) # entity mark
                total += mark
    return total