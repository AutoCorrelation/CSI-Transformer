"""
H5 파일을 읽어 PyTorch 학습용 데이터셋을 구축하는 DataLoader 클래스
"""

import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader, random_split


class DataLoader:
    """
    H5 파일에서 데이터를 로드하고 전처리하여 PyTorch DataLoader를 생성
    
    Parameters:
        h5_path (str): H5 파일 경로
        h5_key (str): H5 파일 내 데이터셋 키 이름 (None이면 첫 번째 키 사용)
        batch_size (int): 배치 크기 (기본값: 64)
        target_shape (tuple): 변경할 Shape (기본값: None - 원본 유지)
        train_ratio (float): 학습 데이터 비율 (기본값: 0.8)
        valid_ratio (float): 검증 데이터 비율 (기본값: 0.1)
        test_ratio (float): 테스트 데이터 비율 (기본값: 0.1)
        random_seed (int): 재현성을 위한 시드 (기본값: 42)
        shuffle_train (bool): 학습 데이터 셔플 여부 (기본값: True)
        num_workers (int): DataLoader worker 수 (기본값: 0)
    """
    
    def __init__(
        self,
        h5_path,
        h5_key=None,
        batch_size=64,
        target_shape=None,
        train_ratio=0.8,
        valid_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
        shuffle_train=True,
        num_workers=0
    ):
        # 비율 검증
        if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
            raise ValueError(f"비율의 합이 1.0이어야 합니다. 현재: {train_ratio + valid_ratio + test_ratio}")
        
        self.h5_path = h5_path
        self.h5_key = h5_key
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        
        # 데이터 로드 및 전처리
        self.data = self._load_h5_data()
        self.data = self._preprocess_data(self.data)
        
        # 데이터셋 분할
        self.train_dataset, self.valid_dataset, self.test_dataset = self._split_dataset()
        
        # DataLoader 생성
        self.train_loader = self._create_dataloader(self.train_dataset, shuffle=self.shuffle_train)
        self.valid_loader = self._create_dataloader(self.valid_dataset, shuffle=False)
        self.test_loader = self._create_dataloader(self.test_dataset, shuffle=False)
    
    def _load_h5_data(self):
        """H5 파일에서 데이터를 로드"""
        with h5py.File(self.h5_path, 'r') as h5_file:
            # 키가 지정되지 않으면 첫 번째 키 사용
            if self.h5_key is None:
                keys = list(h5_file.keys())
                if len(keys) == 0:
                    raise ValueError("H5 파일에 데이터셋이 없습니다.")
                self.h5_key = keys[0]
                print(f"자동 선택된 키: {self.h5_key}")
            
            if self.h5_key not in h5_file:
                raise KeyError(f"키 '{self.h5_key}'가 H5 파일에 존재하지 않습니다. 사용 가능한 키: {list(h5_file.keys())}")
            
            data = h5_file[self.h5_key][...]
            print(f"원본 데이터 Shape: {data.shape}, dtype: {data.dtype}")
        
        return data
    
    def _preprocess_data(self, data):
        """데이터 전처리: dtype 변환 및 reshape"""
        # float32로 변환
        data = data.astype(np.float32)
        
        # PyTorch 텐서로 변환
        data_tensor = torch.from_numpy(data)
        
        # Shape 변경
        if self.target_shape is not None:
            original_shape = data_tensor.shape
            # 첫 번째 차원(샘플 수)을 제외한 나머지 차원만 변경
            if len(self.target_shape) > 0:
                # target_shape에 -1이 포함되어 있으면 자동으로 계산
                new_shape = (data_tensor.shape[0],) + self.target_shape
            else:
                new_shape = self.target_shape
            
            data_tensor = data_tensor.reshape(new_shape)
            print(f"Shape 변경: {original_shape} -> {data_tensor.shape}")
        
        return data_tensor
    
    def _split_dataset(self):
        """데이터를 Train/Valid/Test로 분할"""
        total_size = len(self.data)
        train_size = int(total_size * self.train_ratio)
        valid_size = int(total_size * self.valid_ratio)
        test_size = total_size - train_size - valid_size
        
        # random_split을 위한 Generator 생성 (재현성)
        generator = torch.Generator().manual_seed(self.random_seed)
        
        # TensorDataset 생성
        dataset = TensorDataset(self.data)
        
        # 데이터 분할
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset,
            [train_size, valid_size, test_size],
            generator=generator
        )
        
        print(f"데이터 분할 완료:")
        print(f"  - Train: {len(train_dataset)} 샘플")
        print(f"  - Valid: {len(valid_dataset)} 샘플")
        print(f"  - Test: {len(test_dataset)} 샘플")
        
        return train_dataset, valid_dataset, test_dataset
    
    def _create_dataloader(self, dataset, shuffle=False):
        """Dataset으로부터 DataLoader 생성"""
        return TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_loaders(self):
        """Train, Valid, Test DataLoader 반환"""
        return self.train_loader, self.valid_loader, self.test_loader


# 사용 예시
if __name__ == "__main__":
    # 예시: H5 파일 로드 및 DataLoader 생성
    data_loader = DataLoader(
        h5_path='../data/Real Channel 2D -5dB (C-SRS 5, B-SRS 0, Symbol 4, Comb 4, Tap 4, Slot 1, Correlation 0.90000).h5',
        h5_key=None,  # 또는 None으로 자동 선택
        batch_size=64,
        target_shape=(1, 32, 32),  # (10000, 1024) -> (10000, 1, 32, 32)
        train_ratio=0.8,
        valid_ratio=0.1,
        test_ratio=0.1,
        random_seed=42
    )
    
    train_loader, valid_loader, test_loader = data_loader.get_loaders()
    # 
    # # 학습 루프 예시
    # for batch_idx, (data,) in enumerate(train_loader):
    #     print(f"Batch {batch_idx}: {data.shape}")
    #     break
    
    pass
