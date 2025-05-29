# 修复 XML 标签的脚本
with open('calulate_action_prob.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换标签 - 使用字符串拼接避免自动修改
old_open_tag = "<" + "o" + ">"
new_open_tag = "<" + "Output" + ">"
old_close_tag = "</" + "o" + ">"
new_close_tag = "</" + "Output" + ">"

content = content.replace(old_open_tag, new_open_tag)
content = content.replace(old_close_tag, new_close_tag)

with open('calulate_action_prob.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f'标签替换完成: 打开标签和关闭标签已更新') 