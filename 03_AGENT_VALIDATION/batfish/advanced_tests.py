from pybatfish.client.commands import *
from pybatfish.question import bfq
from pybatfish.datamodel import HeaderConstraints, PathConstraints

class AdvancedValidator:
    
    @staticmethod
    def check_route_exists(network, next_hop=None):
        """Vérifier qu'une route existe"""
        routes = bfq.routes(network=network).answer().frame()
        if next_hop:
            routes = routes[routes['Next_Hop_IP'] == next_hop]
        return not routes.empty
    
    @staticmethod
    def check_connectivity(src, dst):
        """Vérifier la connectivité entre deux équipements"""
        try:
            reachability = bfq.reachability(
                pathConstraints=PathConstraints(startLocation=src),
                headers=HeaderConstraints(dstIps=dst)
            ).answer().frame()
            return not reachability.empty
        except Exception as e:
            print(f"Erreur lors du test de connectivité: {e}")
            return False
    
    @staticmethod
    def check_routing_loops():
        """Détecter les boucles de routage"""
        loops = bfq.detectLoops().answer().frame()
        return loops.empty  # True = pas de boucles
    
    @staticmethod
    def check_bgp_sessions():
        """Vérifier l'état des sessions BGP"""
        bgp = bfq.bgpSessionStatus().answer().frame()
        return bgp
    
    @staticmethod
    def check_unused_structures():
        """Trouver les structures non utilisées dans la config"""
        unused = bfq.unusedStructures().answer().frame()
        return unused

# Test
if __name__ == "__main__":
    bf_session.host = 'localhost'
    bf_init_snapshot('configs/', name='test', overwrite=True)
    
    validator = AdvancedValidator()
    
    print("=== Test de boucles ===")
    no_loops = validator.check_routing_loops()
    print(f"Pas de boucles: {no_loops}")
    
    print("\n=== Structures non utilisées ===")
    unused = validator.check_unused_structures()
    print(unused)