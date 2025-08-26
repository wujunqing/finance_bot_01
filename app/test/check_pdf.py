import os
import fitz  # PyMuPDF
import sys

def check_pdf_files(pdf_dir):
    """检查PDF文件的完整性"""
    if not os.path.exists(pdf_dir):
        print(f"目录不存在: {pdf_dir}")
        return [], []
    
    valid_files = []
    invalid_files = []
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    print(f"找到 {len(pdf_files)} 个PDF文件")
    
    for i, filename in enumerate(pdf_files, 1):
        file_path = os.path.join(pdf_dir, filename)
        try:
            doc = fitz.open(file_path)
            # 尝试读取第一页来验证文件完整性
            if len(doc) > 0:
                page = doc[0]
                text = page.get_text()
            doc.close()
            valid_files.append(file_path)
            print(f"[{i}/{len(pdf_files)}] ✓ 有效: {filename}")
        except Exception as e:
            invalid_files.append((file_path, str(e)))
            print(f"[{i}/{len(pdf_files)}] ✗ 无效: {filename} - {e}")
    
    print(f"\n检查完成:")
    print(f"- 有效文件: {len(valid_files)} 个")
    print(f"- 无效文件: {len(invalid_files)} 个")
    
    if invalid_files:
        print("\n无效文件列表:")
        for file_path, error in invalid_files:
            print(f"  - {os.path.basename(file_path)}: {error}")
    
    return valid_files, invalid_files

def move_invalid_files(invalid_files, backup_dir="dataset/pdf_backup"):
    """将无效文件移动到备份目录"""
    if not invalid_files:
        return
    
    os.makedirs(backup_dir, exist_ok=True)
    
    for file_path, error in invalid_files:
        filename = os.path.basename(file_path)
        backup_path = os.path.join(backup_dir, filename)
        try:
            os.rename(file_path, backup_path)
            print(f"已移动无效文件: {filename} -> {backup_dir}")
        except Exception as e:
            print(f"移动文件失败: {filename} - {e}")

if __name__ == "__main__":
    pdf_dir = "dataset/pdf"
    
    print("开始检查PDF文件...")
    valid_files, invalid_files = check_pdf_files(pdf_dir)
    
    if invalid_files:
        response = input(f"\n发现 {len(invalid_files)} 个无效文件，是否移动到备份目录？(y/n): ")
        if response.lower() == 'y':
            move_invalid_files(invalid_files)
            print("\n无效文件已移动，现在可以重新运行导入脚本。")
    else:
        print("\n所有PDF文件都是有效的。错误可能由其他原因引起。")