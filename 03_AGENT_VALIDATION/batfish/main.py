from validator import ConfigValidator
from advanced_tests import AdvancedValidator

def main():
    print("=== Démarrage de la validation ===\n")
    
    # 1. Charger vos configs existantes
    validator = ConfigValidator()
    validator.load_configs()
    print("✓ Configurations chargées\n")
    
    # 2. Simuler une réponse du LLM
    llm_response = """
hostname R3
!
interface GigabitEthernet0/0
 ip address 192.168.1.3 255.255.255.0
 no shutdown
!
router ospf 1
 network 192.168.1.0 0.0.0.255 area 0
"""
    
    print("=== Configuration suggérée par le LLM ===")
    print(llm_response)
    print()
    
    # 3. Valider la réponse du LLM
    results = validator.validate_llm_response(llm_response, 'R3.cfg')
    
    print("=== Résultats de validation ===")
    if results['valid']:
        print("✓ Configuration VALIDE")
    else:
        print("✗ Configuration INVALIDE")
    
    if results['errors']:
        print("\nErreurs:")
        for error in results['errors']:
            print(f"  - {error}")
    
    if results['warnings']:
        print("\nAvertissements:")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
    # 4. Tests avancés (optionnel)
    print("\n=== Tests avancés ===")
    adv_validator = AdvancedValidator()
    
    no_loops = adv_validator.check_routing_loops()
    print(f"Pas de boucles de routage: {no_loops}")

if __name__ == "__main__":
    main()