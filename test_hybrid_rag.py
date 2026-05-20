"""
测试混合检索演示脚本

运行方式:
    cd /home/l2140/lvyou_harness
    python test_hybrid_rag.py
"""

from rag_adapter import LvyouRAGAdapter

def test_hybrid_retrieval():
    """测试混合检索"""
    print("=" * 60)
    print("测试 BM25 + 向量 + 重排 混合检索")
    print("=" * 60)
    
    # 创建适配器（启用混合检索）
    adapter = LvyouRAGAdapter(
        collection_name="lvyou_guilin",
        top_k=5,
        use_hybrid=True,
    )
    
    # 初始化
    print("\n[1] 初始化 RAG 适配器...")
    success = adapter.initialize()
    print(f"    初始化结果: {'成功' if success else '失败'}")
    
    # 测试检索
    print("\n[2] 测试混合检索...")
    query = "桂林有哪些著名景点"
    
    try:
        results = adapter.retrieve(query, top_k=5, use_hybrid=True)
        print(f"    查询: {query}")
        print(f"    结果数: {len(results)}")
        
        for i, r in enumerate(results, 1):
            print(f"\n    结果 {i}:")
            print(f"      ID: {r.get('id', 'N/A')}")
            print(f"      内容: {r.get('content', '')[:100]}...")
            print(f"      分数: {r.get('score', 0):.4f}")
    except Exception as e:
        print(f"    检索失败: {e}")
    
    # 测试普通检索（回退方案）
    print("\n[3] 测试普通向量检索...")
    try:
        results = adapter.retrieve(query, top_k=5, use_hybrid=False)
        print(f"    查询: {query}")
        print(f"    结果数: {len(results)}")
    except Exception as e:
        print(f"    检索失败: {e}")
    
    # 测试问答
    print("\n[4] 测试问答...")
    try:
        result = adapter.ask("桂林有什么好吃的？")
        print(f"    问题: 桂林有什么好吃的？")
        print(f"    回答: {result.get('answer', '')[:200]}...")
        print(f"    置信度: {result.get('confidence', 0):.2f}")
    except Exception as e:
        print(f"    问答失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_hybrid_retrieval()
