"""Guard the deployment contract: frozen assets travel with the function."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]

def test_frozen_weights_match_the_deployed_model():
    expected = {
        'weights.bin': '90a57d0e7333f9ef473d0b30d7ad56babfbf7db708635bc4879a430f1005d1c6',
        'weights.json': '08031388d94713792ef46f90221f7e066c61501f8cc08aafc3c9414fa3fca820',
    }
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / 'supabase/model' / name).read_bytes()).hexdigest() == digest
