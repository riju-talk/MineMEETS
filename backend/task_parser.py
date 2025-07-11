# task_parser.py
class TaskExtractor:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
    def extract_tasks(self):
        prompt = """
        Extract all tasks from the meeting with owners and deadlines.
        
        Format as JSON:
        [
          {
            "task": "string",
            "owner": "string",
            "deadline": "YYYY-MM-DD"
          }
        ]
        """
        raw_output = self.analyzer.query(prompt)
        # Add validation logic here
        return self._parse_json(raw_output)