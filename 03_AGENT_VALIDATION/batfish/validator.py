from pybatfish.client.commands import *
from pybatfish.question import bfq
import os

class ConfigValidator:
    def __init__(self, config_dir='configs/'):
        self.config_dir = config_dir
        bf_session.host = 'localhost'
        
    def load_configs(self):
        """Charge les configurations dans Batfish"""
        bf_init_snapshot(self.config_dir, name='snapshot', overwrite=True)
    
    def validate_llm_response(self, llm_config_suggestion, filename='temp_config.cfg'):
        """
        Valide la suggestion de configuration du LLM
        
        Args:
            llm_config_suggestion: La configuration suggérée par le LLM
            filename: Nom du fichier temporaire
            
        Returns:
            dict: Résultats de validation
        """
        # Sauvegarder temporairement la config suggérée
        temp_path = os.path.join(self.config_dir, filename)
        with open(temp_path, 'w') as f:
            f.write(llm_config_suggestion)
        
        # Recharger avec la nouvelle config
        self.load_configs()
        
        results = {
            'errors': [],
            'warnings': [],
            'valid': True
        }
        
        # Vérifier les erreurs de syntaxe
        parse_warnings = bfq.parseWarning().answer().frame()
        if not parse_warnings.empty:
            results['errors'].extend(parse_warnings['Text'].tolist())
            results['valid'] = False
        
        # Vérifier les références non définies
        undefined_refs = bfq.undefinedReferences().answer().frame()
        if not undefined_refs.empty:
            results['warnings'].extend(undefined_refs['Lines'].tolist())
        
        # Vérifier la connectivité (si applicable)
        try:
            reachability = bfq.reachability().answer().frame()
            results['reachability_ok'] = not reachability.empty
        except Exception as e:
            results['warnings'].append(f"Reachability check failed: {str(e)}")
        
        return results

# Test si vous exécutez ce fichier directement
if __name__ == "__main__":
    validator = ConfigValidator()
    validator.load_configs()
    
    # Exemple : tester une config générée par le LLM
    llm_output = """
interface GigabitEthernet0/0
 ip address 192.168.1.1 255.255.255.0
 no shutdown
"""
    
    result = validator.validate_llm_response(llm_output)
    print("Résultat de validation:", result)