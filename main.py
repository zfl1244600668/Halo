import os
import re
import logging
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Set, Dict, Generator, List, Optional
import boto3
from botocore.exceptions import NoCredentialsError, ClientError


class S3ImageCleaner:
    def __init__(self, config: Dict):
        self._validate_config(config)
        self.config = config
        self.s3_client = self._init_s3_client()
        self.image_extensions = {
            ext.lower() for ext in config.get(
                'image_extensions', ['jpg', 'jpeg',
                                     'png', 'gif', 'webp', 'bmp']
            )
        }
        self.logger = self._init_logger()
        self._executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)
        # 确保备份路径存在
        os.makedirs(config['backup_path'], exist_ok=True)

    def _validate_config(self, config: Dict):
        """验证必需的配置参数"""
        required_keys = {
            'root_path', 'bucket_name',
            's3_endpoint', 'access_key', 'secret_key', 'backup_path'
        }
        if missing := required_keys - config.keys():
            raise ValueError(f"缺少必要配置参数: {missing}")

    def _init_logger(self) -> logging.Logger:
        """初始化并配置日志记录器"""
        logger = logging.getLogger(self.__class__.__name__)

        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.propagate = False  # 新增这行，阻止日志向上传播到根logger
        return logger

    def _init_s3_client(self):
        """初始化S3客户端连接"""
        return boto3.client(
            's3',
            endpoint_url=self.config['s3_endpoint'],
            aws_access_key_id=self.config['access_key'],
            aws_secret_access_key=self.config['secret_key'],
            region_name=self.config.get('region', 'cn-east-1'),
            config=boto3.session.Config(
                connect_timeout=15,
                read_timeout=30,
                retries={'max_attempts': 3}
            )
        )

    def _get_paginated_s3_objects(self) -> Generator[Dict, None, None]:
        """生成器：分页获取S3存储桶中的所有对象"""
        paginator = self.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=self.config['bucket_name'],
            PaginationConfig={'PageSize': 1000}
        )

        for page in page_iterator:
            if 'Contents' in page:
                yield from page['Contents']

    def get_s3_images(self) -> Set[str]:
        """并行获取S3存储桶中的所有图片文件"""
        try:
            with ThreadPoolExecutor() as executor:
                futures = []
                for obj in self._get_paginated_s3_objects():
                    futures.append(executor.submit(self._is_image_object, obj))

                return {
                    obj['Key'] for future in as_completed(futures)
                    if (obj := future.result()) is not None
                }
        except ClientError as e:
            self.logger.error(f"S3列表查询错误: {e.response['Error']['Message']}")
            return set()

    def _is_image_object(self, obj: Dict) -> Optional[Dict]:
        """判断S3对象是否为图片文件"""
        key = obj['Key']
        extension = key.split('.')[-1].lower()
        if extension in self.image_extensions:
            self.logger.debug(f"Identified S3 image object: {key}")  # 添加日志
            return obj
        return None

    @lru_cache(maxsize=2048)
    def _read_file_cached(self, file_path: str) -> str:
        """带缓存的文件读取（注意：文件修改后需清除缓存）"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            self.logger.warning(f"文件读取错误 {file_path}: {str(e)}")
            return ''

    @classmethod
    def extract_referenced_images(cls, content: str) -> Set[str]:
        """从文本内容中提取所有引用的图片文件名"""
        patterns = [
            # Markdown图片语法
            r'!\[.*?\]\([\'"]?([^\s\?\#]+)[^\s]*[\'"]?\)',
            # HTML图片标签
            r'<img\s+[^>]*src=["\']([^"\'\?#]+)[^"\']*["\']',
            # HTML source标签
            r'<source\s+[^>]*src=["\']([^"\'\?#]+)[^"\']*["\']'
        ]

        results = set()
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                # URL解码并清理路径
                decoded = unquote(match)
                clean_path = os.path.basename(
                    decoded.split('?')[0].split('#')[0]
                )
                if clean_path:
                    results.add(clean_path)
        return results

    def get_local_references(self) -> Set[str]:
        """并行获取本地所有被引用的图片"""
        md_files = []
        for root, _, files in os.walk(self.config['root_path']):
            for file in files:
                if file.lower().endswith('.md'):
                    md_files.append(os.path.join(root, file))

        referenced = set()
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(
                self._process_md_file, f): f for f in md_files}
            for future in as_completed(futures):
                try:
                    referenced.update(future.result())
                except Exception as e:
                    self.logger.warning(f"文件处理错误: {str(e)}")

        return referenced

    def _process_md_file(self, file_path: str) -> Set[str]:
        """处理单个Markdown文件"""
        content = self._read_file_cached(file_path)
        referenced = self.extract_referenced_images(content)
        if referenced:
            self.logger.debug(f"在文件 {file_path.replace('\\', '/')} 中发现引用图片: {referenced}")
        else:
            self.logger.debug(f"在文件 {file_path.replace('\\', '/')} 中未发现任何图片引用")
        return referenced

    def clean_orphan_images(self):
        """执行清理操作（带安全检查）"""
        try:
            self.logger.info("开始清理孤立图片...")

            # 并行获取S3图片和本地引用
            with ThreadPoolExecutor(max_workers=2) as executor:
                s3_future = executor.submit(self.get_s3_images)
                ref_future = executor.submit(self.get_local_references)

                s3_images = s3_future.result()
                referenced_images = ref_future.result()
                self.logger.info(f"S3存储桶中图片数量: {len(s3_images)}")
                self.logger.info(f"本地引用的图片数量: {len(referenced_images)}")

            if not s3_images:
                self.logger.warning("S3存储桶中未找到图片")
                return

            orphan_images = s3_images - referenced_images

            if not orphan_images:
                self.logger.info("未发现孤立图片")
                return

            self.logger.info(f"发现{len(orphan_images)}个待处理孤立图片")
            self._handle_orphan_images(orphan_images)

        except NoCredentialsError:
            self.logger.error("AWS凭证无效或缺失")
        except KeyboardInterrupt:
            self.logger.warning("\n操作被用户中断")
        except Exception as e:
            self.logger.error(f"意外错误: {str(e)}", exc_info=True)
        finally:
            self._executor.shutdown(wait=True)

    def _handle_orphan_images(self, orphan_images: Set[str]):
        """处理孤立图片删除（带确认机制）"""
        sorted_orphans = sorted(orphan_images)

        # 预览前10个文件
        self.logger.info("孤立图片示例（前10个）:")
        for img in sorted_orphans[:10]:
            self.logger.info(f"{img}")

        if self.config.get('dry_run', True):
            self.logger.info("试运行模式：不会实际删除文件")
            return

        # 用户确认
        confirmation = input(f"确认删除{len(orphan_images)}个图片？[y/n]: ")
        if confirmation.lower() != 'y':
            self.logger.info("取消删除操作")
            return

        # 备份孤立图片
        for img in sorted_orphans:
            self.logger.info(f"备份图片: {img}")
            self._backup_image(img)

        # 分批删除（S3单次最多删除1000个对象）
        batch_size = 1000
        for i in range(0, len(sorted_orphans), batch_size):
            batch = sorted_orphans[i:i + batch_size]
            self._delete_batch(batch)

    def _backup_image(self, image_key: str):
        """备份图片到本地"""
        try:
            image_path = os.path.join(
                self.config['backup_path'], os.path.basename(image_key)
            )
            self.s3_client.download_file(
                self.config['bucket_name'], image_key, image_path
            )
        except Exception as e:
            self.logger.error(f"备份图片失败 {image_key}: {str(e)}")

    def _delete_batch(self, batch: List[str]):
        """批量删除S3对象"""
        try:
            response = self.s3_client.delete_objects(
                Bucket=self.config['bucket_name'],
                Delete={'Objects': [{'Key': key} for key in batch]}
            )
            if 'Errors' in response:
                for error in response['Errors']:
                    self.logger.error(f"删除失败 {error['Key']}: {
                                      error['Message']}")
            self.logger.info(
                f"成功删除 {len(batch) - len(response.get('Errors', []))} 个图片")
        except ClientError as e:
            self.logger.error(f"批量删除失败: {e.response['Error']['Message']}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._executor.shutdown(wait=True)


def main():
    # 配置示例（实际使用时建议通过环境变量管理敏感信息）
    config = {
        'root_path': "C:/Users/zfl18/Downloads/test",
        'backup_path': "C:/Users/zfl18/Downloads/test/test_新",
        'bucket_name': "pic-aoliaoduo",
        's3_endpoint': "https://s3.bitiful.net",
        'access_key': "lGzpnQKxnux7G4sOgupoqfag",
        'secret_key': "asRhdjHDV7pjN9t2fT0FizxEluF1khT",
        'dry_run': False,  # 试运行模式开关
        'image_extensions': ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp']
    }

    with S3ImageCleaner(config) as cleaner:
        cleaner.clean_orphan_images()


if __name__ == "__main__":
    main()
