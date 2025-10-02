from argparse import Namespace
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import Dataset
import dgl
from dgl.dataloading import GraphDataLoader
from rdkit.Chem.BRICS import FindBRICSBonds, BreakBRICSBonds
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import MACCSkeys
from rdkit import RDConfig
from rdkit import RDLogger

from features.featurization import brics_features, GetBRICSBondFeature_Hetero

RDLogger.DisableLog('rdApp.*')  
import os


fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

def bond_features(bond: Chem.rdchem.Bond):
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))

    return fbond

def pharm_property_types_feats(mol,factory=factory): 
    types = [i.split('.')[1] for i in factory.GetFeatureDefs().keys()]
    feats = [i.GetType() for i in factory.GetFeaturesForMol(mol)]
    result = [0] * len(types)
    for i in range(len(types)):
        if types[i] in list(set(feats)):
            result[i] = 1
    return result

def GetBricsBonds(mol):
    brics_bonds = list()
    brics_bonds_rules = list()
    #找到所有的BRICS键[(0,1),(2,3)] 表示 0 1 以及 2 3这两个化学键
    bonds_tmp = FindBRICSBonds(mol)
    #[((1, 3), ('1', '3')), ((3, 4), ('3', '16'))] (1,3)是以前的化学键 ('1','3')是表示 BRICS 类型（由两个 BRICS 类型标识符组成的元组）
    bonds = [b for b in bonds_tmp]
    for item in bonds:# item[0] is bond, item[1] is brics type
        brics_bonds.append([int(item[0][0]), int(item[0][1])])
        # 得到该BRICS键的特征向量，于是brics_bonds_rules为 [[[3, 1], 特征向量].........]
        brics_bonds_rules.append([[int(item[0][0]), int(item[0][1])], GetBricsBondFeature([item[1][0], item[1][1]])])
        brics_bonds.append([int(item[0][1]), int(item[0][0])])
        brics_bonds_rules.append([[int(item[0][1]), int(item[0][0])], GetBricsBondFeature([item[1][1], item[1][0]])])

    result = []
    #这段代码中主要的目的是，如果一个mol中含有多个相同的BRICS键，那么就多个一起加入result中，比如如果含有2个C-O键，那么两个都会加入到result
    for bond in mol.GetBonds():
        beginatom = bond.GetBeginAtomIdx()
        endatom = bond.GetEndAtomIdx()
        if [beginatom, endatom] in brics_bonds:
            result.append([bond.GetIdx(), beginatom, endatom])
            
    return result, brics_bonds_rules

def GetBricsBondFeature(action):
    result = []
    start_action_bond = int(action[0]) if (action[0] !='7a' and action[0] !='7b') else 7
    end_action_bond = int(action[1]) if (action[1] !='7a' and action[1] !='7b') else 7
    emb_0 = [0 for i in range(17)]
    emb_1 = [0 for i in range(17)]
    emb_0[start_action_bond] = 1
    emb_1[end_action_bond] = 1
    result = emb_0 + emb_1
    return result

def maccskeys_emb(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()+1) # aviod index 0
    return mol

def GetFragmentFeats(mol):
    #得到BRICS键的ID号，比如有可能是该分子中的C-O键，得到其的ID号，比如3，  BRICS键是一种特殊的化学键
    break_bonds = [mol.GetBondBetweenAtoms(i[0][0],i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]
    if break_bonds == []:
        tmp = mol
    else:
        #断开BRICS键的化学式，其smiles和没断开的有一点点不同，比如CCO 和 CC.OH
        tmp = Chem.FragmentOnBonds(mol,break_bonds,addDummies=False)
    #得到被断开BRICS键的原子，比如 （（0，1,2），（3）），这两个部分是两个断开分子的原子
    frags_idx_lst = Chem.GetMolFrags(tmp)
    result_ap = {}
    result_p = {}
    pharm_id = 0
    for frag_idx in frags_idx_lst:
        for atom_id in frag_idx:
            result_ap[atom_id] = pharm_id
        try:
            mol_pharm = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, frag_idx))
            #得到片段的分子指纹
            emb_0 = maccskeys_emb(mol_pharm)
            #得到一个向量 ，是药效团特征，如果为1则表示有该特征，0就是没有
            emb_1 = pharm_property_types_feats(mol_pharm)
        except Exception:
            emb_0 = [0 for i in range(167)]
            emb_1 = [0 for i in range(27)]
        #拼接
        result_p[pharm_id] = emb_0 + emb_1

        pharm_id += 1
    return result_ap, result_p

ELEMENTS = [35, 6, 7, 8, 9, 15, 16, 17, 53]
ATOM_FEATURES = {
    'atomic_num': ELEMENTS,
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def atom_features(atom: Chem.rdchem.Atom):
    features = onek_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    return features


def Mol2HeteroGraph(mol):
    
    # build graphs
    # ('a', 'b', 'a')：表示原子之间的键。
    # ('p', 'r', 'p')：表示片段之间的反应连接。
    # ('a', 'j', 'p') 和 ('p', 'j', 'a')：表示原子和片段之间的连接（junction）
    # 定义四种类型的边
    edge_types = [('a','b','a'),('p','r','p'),('a','j','p'), ('p','j','a')]

    edges = {k:[] for k in edge_types}
    # if mol.GetNumAtoms() == 1:
    #     g = dgl.heterograph(edges, num_nodes_dict={'a':1,'p':1})
    # else:
    #result_ap[0] = 1 表示原子编号为0的原子，属于编号1的片段, result_p[1] = [1,2,3,0,4,1...] 表示片段1的特征
    result_ap, result_p = GetFragmentFeats(mol)

    #reac_idx 是 [[1,2,3],....] 1表示BRICS键的编号，2,3表示原子编号，bbr是[[[2,3],[xxxxx]]] bbr含有该键位的特征
    reac_idx, bbr = GetBricsBonds(mol)

    for bond in mol.GetBonds():
        #正反两个键位都加入进去
        edges[('a','b','a')].append([bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()])
        edges[('a','b','a')].append([bond.GetEndAtomIdx(),bond.GetBeginAtomIdx()])

    for r in reac_idx:
        begin = r[1]
        end = r[2]

        #[begin,end]是两个原子之间的键，不过是片段的原子，而result_ap 就将该原子的id映射到了片段的id
        edges[('p','r','p')].append([result_ap[begin],result_ap[end]])
        edges[('p','r','p')].append([result_ap[end],result_ap[begin]])

    for k,v in result_ap.items():
        #将原子id和片段的id一一映射
        edges[('a','j','p')].append([k,v])
        edges[('p','j','a')].append([v,k])

    #对于每一种边，都构建一个邻接矩阵
    g = dgl.heterograph(edges)

    #遍历所有原子节点，提取原子特征并存储在图的节点数据中。
    f_atom = []
    for idx in g.nodes('a'):
        atom = mol.GetAtomWithIdx(idx.item())
        f_atom.append(atom_features(atom))
    f_atom = torch.FloatTensor(f_atom)
    #data是一个字典，f就是一个随便的名字
    g.nodes['a'].data['f'] = f_atom
    dim_atom = len(f_atom[0])

    #遍历片段特征，将其该节点存储在图的片段节点数据中。
    f_pharm = []
    for k,v in result_p.items():
        f_pharm.append(v)
    g.nodes['p'].data['f'] = torch.FloatTensor(f_pharm)
    dim_pharm = len(f_pharm[0])
    
    #为连接特征（junction features）初始化数据，通过将原子和片段特征进行拼接。
    dim_atom_padding = g.nodes['a'].data['f'].size()[0]
    dim_pharm_padding = g.nodes['p'].data['f'].size()[0]

    g.nodes['a'].data['f_junc'] = torch.cat([g.nodes['a'].data['f'], torch.zeros(dim_atom_padding, dim_pharm)], 1)
    g.nodes['p'].data['f_junc'] = torch.cat([torch.zeros(dim_pharm_padding, dim_atom), g.nodes['p'].data['f']], 1)
    
    # features of edges

    #遍历所有键边，提取键特征并存储在图的边数据中。

    f_bond = []
    src,dst = g.edges(etype=('a','b','a'))
    for i in range(g.num_edges(etype=('a','b','a'))):
        f_bond.append(bond_features(mol.GetBondBetweenAtoms(src[i].item(),dst[i].item())))
    g.edges[('a','b','a')].data['x'] = torch.FloatTensor(f_bond)
    # 遍历所有反应边，提取反应特征并存储在图的边数据中。
    f_reac = []



    src, dst = g.edges(etype=('p','r','p'))
    for idx in range(g.num_edges(etype=('p','r','p'))):
        p0_g = src[idx].item()
        p1_g = dst[idx].item()
        for i in bbr:
            p0 = result_ap[i[0][0]]
            p1 = result_ap[i[0][1]]
            if p0_g == p0 and p1_g == p1:
                f_reac.append(i[1])
    if len(f_reac) == 0:  # 新增空数据保护
        g.edges[('p', 'r', 'p')].data['x'] = torch.zeros((0, 34), dtype=torch.float32)
    else:
        g.edges[('p', 'r', 'p')].data['x'] = torch.FloatTensor(f_reac)


    return g
def Mol2HeteroGraph2(mol):
    # build graphs
    edge_types = [('a', 'b', 'a'), ('p', 'r', 'p'), ('a', 'j', 'p'), ('p', 'j', 'a')]
    pharm_feats, atom2pharmid, frags_idx_lst = brics_features(mol, pretrain=False)
    result = GetBricsBonds(mol)
    edges = {k: [] for k in edge_types}
    for bond in mol.GetBonds():
        edges[('a', 'b', 'a')].append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edges[('a', 'b', 'a')].append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    for (a1, a2) in result.keys():
        edges[('p', 'r', 'p')].append([atom2pharmid[a1], atom2pharmid[a2]])

    for k, v in atom2pharmid.items():
        edges[('a', 'j', 'p')].append([k, v])
        edges[('p', 'j', 'a')].append([v, k])
    g = dgl.heterograph(edges)
    # atom view
    f_atom = []
    for idx in g.nodes('a'):
        atom = mol.GetAtomWithIdx(idx.item())
        f_atom.append(atom_features(atom))

    f_bond = []
    src, dst = g.edges(etype=('a', 'b', 'a'))
    for i in range(g.num_edges(etype=('a', 'b', 'a'))):

        f_bond.append(f_atom[src[i].item()] + bond_features(mol.GetBondBetweenAtoms(src[i].item(), dst[i].item())))

    g.edges[('a', 'b', 'a')].data['x'] = torch.FloatTensor(f_bond)
    f_atom = torch.FloatTensor(f_atom)
    g.nodes['a'].data['f'] = f_atom
    dim_atom = len(f_atom[0])

    # pharm view
    f_pharm = []
    for k, v in pharm_feats.items():
        f_pharm.append(v)

    f_reac = []
    src, dst = g.edges(etype=('p', 'r', 'p'))
    for idx in range(g.num_edges(etype=('p', 'r', 'p'))):
        p1 = src[idx].item()
        p2 = dst[idx].item()
        for k, v in result.items():
            if p1 == atom2pharmid[k[0]] and p2 == atom2pharmid[k[1]]:

                f_reac.append(f_pharm[p1] + GetBRICSBondFeature_Hetero(v[0], v[1]))


    g.edges[('p', 'r', 'p')].data['x'] = torch.FloatTensor(f_reac)
    g.nodes['p'].data['f'] = torch.FloatTensor(f_pharm)
    dim_pharm = len(f_pharm[0])

    dim_atom_padding = g.nodes['a'].data['f'].size()[0]  # 原子个数
    dim_pharm_padding = g.nodes['p'].data['f'].size()[0]  # 药效团个数
    # junction view
    g.nodes['a'].data['f_junc'] = torch.cat([g.nodes['a'].data['f'], torch.zeros(dim_atom_padding, dim_pharm)], 1)
    g.nodes['p'].data['f_junc'] = torch.cat([torch.zeros(dim_pharm_padding, dim_atom), g.nodes['p'].data['f']], 1)

    return g


def Mol2AtomGraph2(mol):
    # build graphs
    edge_types = [('a', 'b', 'a')]

    result = GetBricsBonds(mol)
    edges = {k: [] for k in edge_types}
    for bond in mol.GetBonds():
        edges[('a', 'b', 'a')].append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edges[('a', 'b', 'a')].append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])

    g = dgl.heterograph(edges)
    # atom view
    f_atom = []
    for idx in g.nodes('a'):
        atom = mol.GetAtomWithIdx(idx.item())
        f_atom.append(atom_features(atom))

    f_bond = []
    src, dst = g.edges(etype=('a', 'b', 'a'))
    for i in range(g.num_edges(etype=('a', 'b', 'a'))):
        f_bond.append(f_atom[src[i].item()] + bond_features(mol.GetBondBetweenAtoms(src[i].item(), dst[i].item())))

    g.edges[('a', 'b', 'a')].data['x'] = torch.FloatTensor(f_bond)
    f_atom = torch.FloatTensor(f_atom)
    g.nodes['a'].data['f'] = f_atom


    return g

def Mol2AtomGraph(mol):
    edge_types = [('a', 'b', 'a')]

    edges = {k: [] for k in edge_types}

    for bond in mol.GetBonds():
        # 正反两个键位都加入进去
        edges[('a', 'b', 'a')].append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edges[('a', 'b', 'a')].append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])

    # 对于每一种边，都构建一个邻接矩阵
    g = dgl.heterograph(edges)

    # 遍历所有原子节点，提取原子特征并存储在图的节点数据中。
    f_atom = []
    for idx in g.nodes('a'):
        atom = mol.GetAtomWithIdx(idx.item())
        f_atom.append(atom_features(atom))
    f_atom = torch.FloatTensor(f_atom)
    # data是一个字典，f就是一个随便的名字
    g.nodes['a'].data['f'] = f_atom

    # features of edges

    # 遍历所有键边，提取键特征并存储在图的边数据中。

    f_bond = []
    src, dst = g.edges(etype=('a', 'b', 'a'))
    for i in range(g.num_edges(etype=('a', 'b', 'a'))):
        f_bond.append(bond_features(mol.GetBondBetweenAtoms(src[i].item(), dst[i].item())))
    g.edges[('a', 'b', 'a')].data['x'] = torch.FloatTensor(f_bond)


    return g

#MolGraphSet类含有self.mols（原子图）、self.graphs（原子-片段的异构图）、self.labels（标签）三个属性。
# 该集合会返回self.graphs[idx] 与 self.labels[idx]
#    def __getitem__(self,idx):
#        return self.graphs[idx],self.labels[idx]
class MolGraphSet(Dataset):
    def __init__(self,df,target,log=print):
        self.data = df
        self.mols = []

        self.labels = []
        self.graphs = []
        for i,row in df.iterrows():
            smi = row['smiles']
            label = row[target].values.astype(float)
            try:
                print("smi is ",smi)
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    log('invalid',smi)
                else:
                    g = Mol2HeteroGraph(mol)
                    if g.num_nodes('a') == 0:
                        log('no edge in graph',smi)
                    else:
                        self.mols.append(mol)
                        self.graphs.append(g)
                        self.labels.append(label)
            except Exception as e:
                log(e,'invalid',smi)
                
    def __len__(self):
        return len(self.mols)
    
    def __getitem__(self,idx):
        
        return self.graphs[idx],self.labels[idx]





