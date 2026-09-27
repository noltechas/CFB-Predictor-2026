import unittest
import torch
from cfbpred.ensemble import align_vocabulary

class FrozenVocabularyTests(unittest.TestCase):
    def test_removed_team_does_not_shift_other_embeddings(self):
        saved={'team':{'Alabama':1,'Removed school':2,'Maine':3},'conf':{'SEC':1,'CAA':2},'n_hist':2}
        d={'vocab':{'team':{'Maine':1,'Alabama':2,'New school':3},'conf':{'CAA':1,'SEC':2},'n_hist':2},
           'G_hidx':torch.tensor([1,2,3]),'G_aidx':torch.tensor([2,1,0]),
           'G_opp':torch.tensor([[[1,2],[3,0]]]),'G_hconf':torch.tensor([1,2,0]),'G_aconf':torch.tensor([2,1,0])}
        embeddings=torch.tensor([0.,10.,20.,30.])
        align_vocabulary(d,saved)
        self.assertEqual(embeddings[d['G_hidx']].tolist(),[30.,10.,0.])
        self.assertEqual(d['G_opp'].tolist(),[[[3,1],[0,0]]])
        self.assertEqual(d['G_hconf'].tolist(),[2,1,0])
        before={k:v.clone() for k,v in d.items() if torch.is_tensor(v)}
        align_vocabulary(d,saved)
        self.assertTrue(all(torch.equal(v,d[k]) for k,v in before.items()))
