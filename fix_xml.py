#!/usr/bin/env python3
# 修复 XML 标签

file_path = 'calulate_action_prob.py'

# 读取文件的所有行
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 修改目标行
for i in range(len(lines)):
    # 修改系统提示中的标签
    if '  <o>' in lines[i]:
        lines[i] = lines[i].replace('  <o>', '  <Output>')
    elif '  </o>' in lines[i]:
        lines[i] = lines[i].replace('  </o>', '  </Output>')

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('XML 标签替换完成！') 