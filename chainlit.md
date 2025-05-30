# Welcome to Trend Cybertron - Cloud Risk Assessment Agent! 🚀🤖

Hi there! 👋  
I am a security scan reasoning agent designed to help you understand your security scan reports and provide valuable insights.  

Note: The misconfiguration CVSS scores for Kubernetes and AWS were generated by an LLM based on the CVSS scoring system.

## 🔹 Useful Commands

### 📊 Generate an Executive Summary  
Use the following commands to generate executive summaries based on your security scan type:

```
# Use these commands or the buttons under the initial chat screen
/report all
/report code
/report container
/report aws
/report kubernetes
```

## Best Practices for Security Scan Questions

### 1. Specify Scan Category
Choose the appropriate security scan type:
- Code Analysis
- Container Security
- Kubernetes Configuration
- AWS Infrastructure
- Overall across categories

### 2. Resource Information
Include these details in your queries:
- Resource Type (e.g., EC2 instance, Docker image, K8s deployment)
- Resource Name/ID
- Environment (Dev/Staging/Prod)
- Region (for cloud resources)

### 3. Query Guidelines
For optimal results:
- Be specific about security concerns
- Include relevant context
- Focus on one security issue per query
- Specify compliance requirements (if applicable)

## ❓ Sample Questions  

- What are the top critical issues in the Kubernetes scan report?
- Could you provide detailed information about AVD-AWS-0101?
- Explain what AVD-KSV-0109 is and list my affected resources.
- Give me a short summary of the container report for high-risk issues only.
