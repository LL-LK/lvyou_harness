"""
测试混合检索演示脚本

运行方式:
    cd /home/l2140/lvyou_harness
    PYTHONPATH=/home/l2140 python test_hybrid_rag.py
"""
import sys
sys.path.insert(0, '/home/l2140/lvyou_harness')

from lvyou_harness.adapters.rag_adapter_v2 import MilvusRAGAdapter, MilvusRAGConfig


def test_hybrid_retrieval():
    """测试混合检索（BM25 + 向量 + 重排）"""
    print("=" * 60)
    print("测试 BM25 + 向量 + 重排 混合检索")
    print("=" * 60)

    # 创建适配器
    config = MilvusRAGConfig()
    adapter = MilvusRAGAdapter(config)

    # 初始化
    print("\n[1] 初始化 RAG 适配器...")
    success = adapter.initialize()
    print(f"    初始化结果: {'成功' if success else '失败'}")
    print(f"    文档总数: {adapter.count()}")

    # 测试检索
    print("\n[2] 测试混合检索...")
    query = "桂林有哪些著名景点"

    try:
        results = adapter.retrieve(query, top_k=5)
        print(f"    查询: {query}")
        print(f"    结果数: {len(results)}")

        for i, r in enumerate(results, 1):
            print(f"\n    结果 {i}:")
            print(f"      ID: {r.get('id', 'N/A')}")
            print(f"      标题: {r.get('title', 'N/A')}")
            print(f"      内容: {r.get('content', '')[:100]}...")
            print(f"      分数: {r.get('score', 0):.4f}")
    except Exception as e:
        import traceback
        print(f"    检索失败: {e}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_hybrid_retrieval()
