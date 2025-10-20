from pybatfish.client.commands import *
from pybatfish.question import bfq

# Connexion à Batfish
bf_session.host = 'localhost'
bf_init_snapshot('configs/', name='test_network', overwrite=True)

# Exemple de validations
def validate_configuration():
    # 1. Vérifier les erreurs de parsing
    parse_warnings = bfq.parseWarning().answer().frame()
    print("=== Erreurs de parsing ===")
    print(parse_warnings)
    
    # 2. Vérifier les interfaces définies
    interfaces = bfq.interfaceProperties().answer().frame()
    print("\n=== Interfaces ===")
    print(interfaces[['Interface', 'Active', 'Primary_Address']])
    
    # 3. Vérifier les routes
    routes = bfq.routes().answer().frame()
    print("\n=== Routes ===")
    print(routes)
    
    return True

if __name__ == "__main__":
    validate_configuration()