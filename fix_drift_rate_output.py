#!/usr/bin/env python3
"""
Fix drift rate output issue - Remove ReLU from output layer
"""

import json

# Read notebook
with open('/Users/siyu/Documents/GitHub/VAM-studying/vgg_drift_rate_complete.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find and fix the drift_head definition
for i, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') == 'code' and 'nn.Linear(256, 4),  # 输出4个漂移率' in str(cell.get('source', [])):
        old_source = ''.join(cell['source'])
        
        # Fix 1: Remove ReLU from output
        new_source = old_source.replace(
            'nn.Linear(256, 4),  # 输出4个漂移率\n        nn.ReLU()  # 确保漂移率为正值',
            'nn.Linear(256, 4)  # 输出4个漂移率 (不使用ReLU，允许负值)'
        )
        
        # Fix 2: Add positive constraint using softplus
        new_source = new_source.replace(
            'def forward(self, x):',
            '''def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像 (batch_size, 3, 128, 128)
            
        返回:
            drift_rates: 漂移率 (batch_size, 4)
                        [左, 右, 上, 下]
                        使用Softplus确保正值
        """'''
        )
        
        new_source = new_source.replace(
            'drift_rates = self.drift_head(x)  # (batch, 4)\n        \n        return drift_rates',
            'drift_rates = self.drift_head(x)  # (batch, 4)\n        drift_rates = F.softplus(drift_rates)  # 确保漂移率为正值\n        \n        return drift_rates'
        )
        
        cell['source'] = new_source.split('\n')
        print(f"✓ Fixed cell {i+1}: Removed ReLU, added Softplus")
        break

# Save fixed notebook
with open('/Users/siyu/Documents/GitHub/VAM-studying/vgg_drift_rate_complete.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("\n✓ Notebook fixed!")
print("\nChanges:")
print("  1. Removed ReLU from output layer")
print("  2. Added Softplus to ensure positive drift rates")
print("  3. Softplus: log(1 + exp(x)) - smooth approximation of ReLU")
print("\nNote: This fixes the technical issue, but ELBO is still recommended")
print("      for proper cognitive modeling.")
