import requests
import sys
import os

# Test the analyze endpoint with a real docx file
test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'develop_documents', 'CH3.2风险管理模板', '有源医疗器械9706.1-2020风险管理报告.docx')

try:
    with open(test_file, 'rb') as f:
        files = {'file': ('test.docx', f, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')}
        data = {'session_id': 'test', 'question': '请审核这份文件'}
        response = requests.post('http://localhost:8001/api/analyze', files=files, data=data, timeout=120)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Has retrieved_docs: {'retrieved_docs' in result}")
        if 'retrieved_docs' in result:
            print(f"retrieved_docs count: {len(result.get('retrieved_docs', []))}")
            if result.get('retrieved_docs'):
                print(f"First doc source: {result['retrieved_docs'][0].get('source', 'N/A')}")
        print(f"Answer preview: {result.get('answer', '')[:200]}")
except Exception as e:
    print(f"Error: {e}")
