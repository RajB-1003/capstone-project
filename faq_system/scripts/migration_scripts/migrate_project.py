import os
import shutil
import re

def migrate():
    # Base directory is faq_system/
    # The script should be run from d:\Capstone Project
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'faq_system')
    if not os.path.exists(BASE_DIR):
        print(f"Error: {BASE_DIR} does not exist.")
        return

    DIRS_TO_CREATE = [
        'app/ui/components',
        'app/services',
        'app/core',
        'app/storage',
        'app/auth',
        'app/utils',
        'data/db',
        'data/embeddings',
        'data/configs',
        'data/logs',
        'scripts/migration_scripts',
        'tests',
        'config'
    ]

    for d in DIRS_TO_CREATE:
        os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)
        # Create __init__.py in python packages
        if d.startswith('app') or d == 'config' or d == 'tests':
            init_file = os.path.join(BASE_DIR, d, '__init__.py')
            with open(init_file, 'w') as f:
                pass
                
    # Create top level app/__init__.py
    with open(os.path.join(BASE_DIR, 'app', '__init__.py'), 'w') as f:
        pass

    MOVES = [
        ('app.py', 'app/main.py'),
        ('modules/pipeline.py', 'app/services/pipeline.py'),
        ('modules/router_tier1.py', 'app/services/router.py'),
        ('modules/router_tier2.py', 'app/services/router_tier2.py'), # Keep both or merge? Let's just move
        ('modules/retriever.py', 'app/services/retriever.py'),
        ('modules/rag_demo.py', 'app/services/rag_service.py'),
        ('modules/evaluation.py', 'app/services/feedback_service.py'),
        ('modules/embedder.py', 'app/core/embedder.py'),
        ('modules/semantic_search.py', 'app/core/semantic_search.py'),
        ('modules/hybrid_search.py', 'app/core/hybrid_search.py'),
        ('modules/keyword_search.py', 'app/core/keyword_search.py'),
        ('modules/confidence.py', 'app/core/confidence.py'),
        ('modules/db.py', 'app/storage/db.py'),
        ('modules/cache.py', 'app/storage/cache.py'),
        ('modules/feedback_store.py', 'app/storage/feedback_store.py'),
        ('modules/faq_manager.py', 'app/storage/query_store.py'),
        ('modules/embedding_store.py', 'app/storage/embedding_store.py'),
        ('modules/auth.py', 'app/auth/auth.py'),
        ('modules/profiler.py', 'app/utils/profiler.py'),
        ('modules/voice_utils.py', 'app/utils/helpers.py'),
        ('modules/admin_dashboard.py', 'app/ui/admin_dashboard.py'),
        ('modules/comparison.py', 'app/ui/comparison.py'),
        ('modules/multilingual.py', 'app/core/multilingual.py'),
        ('modules/query_filter.py', 'app/core/query_filter.py'),
        ('modules/constants.py', 'config/settings.py'),
        
        # Data files
        ('data/db.sqlite3', 'data/db/db.sqlite3'),
        ('data/corpus_embeddings.npy', 'data/embeddings/corpus_embeddings.npy'),
        ('data/faqs.json', 'data/configs/faqs.json'),
        ('data/intent_exemplars.json', 'data/configs/intent_exemplars.json'),
        ('config/regex_patterns.json', 'data/configs/regex_patterns.json'),
        ('data/feedback_log.jsonl', 'data/logs/feedback_log.jsonl'),
        
        # Scripts
        ('fix_encoding.py', 'scripts/fix_encoding.py'),
        ('validate_phase1.py', 'scripts/validate_phase1.py'),
        ('validate_phase2.py', 'scripts/validate_phase2.py'),
        ('validate_phase3.py', 'scripts/validate_phase3.py'),
        ('validate_phase4.py', 'scripts/validate_phase4.py'),
        ('validate_phase5.py', 'scripts/validate_phase5.py'),
        ('validate_phase6.py', 'scripts/validate_phase6.py'),
        ('validate_phase7.py', 'scripts/validate_phase7.py'),
        ('validate_phase_multilingual.py', 'scripts/validate_phase_multilingual.py'),
        ('validate_multilingual_robustness.py', 'scripts/validate_multilingual_robustness.py'),
        ('modules/install_voice_deps.py', 'scripts/install_voice_deps.py'),
        
        # Any tests
        ('tests/test_phase1.py', 'tests/test_phase1.py'),
    ]

    for old, new in MOVES:
        old_path = os.path.join(BASE_DIR, old)
        new_path = os.path.join(BASE_DIR, new)
        if os.path.exists(old_path):
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(old_path, new_path)
            print(f"Moved: {old} -> {new}")

    # Remove now-empty old dirs
    for d in ['modules', 'config', 'data', 'tests']:
        old_dir = os.path.join(BASE_DIR, d)
        # only delete if empty
        if os.path.exists(old_dir) and not os.listdir(old_dir):
            os.rmdir(old_dir)

    IMPORT_MAPPING = {
        'modules.pipeline': 'app.services.pipeline',
        'modules.router_tier1': 'app.services.router',
        'modules.router_tier2': 'app.services.router_tier2',
        'modules.retriever': 'app.services.retriever',
        'modules.rag_demo': 'app.services.rag_service',
        'modules.evaluation': 'app.services.feedback_service',
        'modules.embedder': 'app.core.embedder',
        'modules.semantic_search': 'app.core.semantic_search',
        'modules.hybrid_search': 'app.core.hybrid_search',
        'modules.keyword_search': 'app.core.keyword_search',
        'modules.confidence': 'app.core.confidence',
        'modules.db': 'app.storage.db',
        'modules.cache': 'app.storage.cache',
        'modules.feedback_store': 'app.storage.feedback_store',
        'modules.faq_manager': 'app.storage.query_store',
        'modules.embedding_store': 'app.storage.embedding_store',
        'modules.auth': 'app.auth.auth',
        'modules.profiler': 'app.utils.profiler',
        'modules.voice_utils': 'app.utils.helpers',
        'modules.admin_dashboard': 'app.ui.admin_dashboard',
        'modules.comparison': 'app.ui.comparison',
        'modules.constants': 'config.settings',
        'modules.multilingual': 'app.core.multilingual',
        'modules.query_filter': 'app.core.query_filter',
    }

    def update_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        for old_mod, new_mod in IMPORT_MAPPING.items():
            content = re.sub(rf'\b{old_mod.replace(".", r"\.")}\b', new_mod, content)
            
        content = content.replace('"data/faqs.json"', '"data/configs/faqs.json"')
        content = content.replace("'data/faqs.json'", "'data/configs/faqs.json'")
        content = content.replace('os.path.join("data", "faqs.json")', 'os.path.join("data", "configs", "faqs.json")')
        
        content = content.replace('"data/intent_exemplars.json"', '"data/configs/intent_exemplars.json"')
        content = content.replace("'data/intent_exemplars.json'", "'data/configs/intent_exemplars.json'")
        content = content.replace('os.path.join("data", "intent_exemplars.json")', 'os.path.join("data", "configs", "intent_exemplars.json")')
        
        content = content.replace('"config/regex_patterns.json"', '"data/configs/regex_patterns.json"')
        content = content.replace("'config/regex_patterns.json'", "'data/configs/regex_patterns.json'")
        content = content.replace('os.path.join("config", "regex_patterns.json")', 'os.path.join("data", "configs", "regex_patterns.json")')

        content = content.replace('"data/corpus_embeddings.npy"', '"data/embeddings/corpus_embeddings.npy"')
        content = content.replace('os.path.join("data", "corpus_embeddings.npy")', 'os.path.join("data", "embeddings", "corpus_embeddings.npy")')
        
        content = content.replace('"data/db.sqlite3"', '"data/db/db.sqlite3"')
        content = content.replace('os.path.join("data", "db.sqlite3")', 'os.path.join("data", "db", "db.sqlite3")')

        content = content.replace('"data/feedback_log.jsonl"', '"data/logs/feedback_log.jsonl"')
        content = content.replace('os.path.join("data", "feedback_log.jsonl")', 'os.path.join("data", "logs", "feedback_log.jsonl")')

        if filepath.endswith('main.py'):
            content = content.replace(
                "sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))",
                "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))"
            )

        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated imports in {os.path.basename(filepath)}")

    for root, _, files in os.walk(BASE_DIR):
        for f in files:
            if f.endswith('.py'):
                update_file(os.path.join(root, f))

    print("\nMigration Complete! Please run:")
    print("cd faq_system")
    print("streamlit run app/main.py")

if __name__ == "__main__":
    migrate()
