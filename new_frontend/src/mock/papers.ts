import type { Paper, RecommendedPaper, LabeledPaper } from '@/types/paper'

export const mockPapers: Paper[] = [
  {
    id: '1',
    arxivId: '2312.16130',
    title: 'LLaMA-3: Open Foundation and Fine-Tuned Chat Models',
    authors: ['Hugo Touvron', 'Louis Martin', 'Kevin Stone', 'Peter Albert'],
    summary: 'We introduce LLaMA-3, a family of open-source large language models (LLMs) designed to be the foundation for future AI research and applications. LLaMA-3 models are trained on a diverse mix of publicly available data, demonstrating strong performance across various benchmarks. Our largest model, LLaMA-3-70B, achieves state-of-the-art results on reasoning, coding, and language understanding tasks while maintaining open access for research purposes.',
    publishedAt: '2023-12-27',
    updatedAt: '2024-01-15',
    categories: ['cs.AI', 'cs.CL'],
    pdfUrl: 'https://arxiv.org/pdf/2312.16130',
    absUrl: 'https://arxiv.org/abs/2312.16130',
    label: 'liked'
  },
  {
    id: '2',
    arxivId: '2401.02990',
    title: 'Vision-Language Models: A Survey and Taxonomy',
    authors: ['Jiwei Li', 'Xia Wang', 'Yong Zhang'],
    summary: 'Vision-language models (VLMs) have emerged as a powerful paradigm for AI systems that can understand both visual and textual information. This survey provides a comprehensive overview of recent advances in VLMs, covering architecture designs, training strategies, and applications. We categorize existing models based on their architectural choices and discuss the challenges and future directions in this rapidly evolving field.',
    publishedAt: '2024-01-05',
    categories: ['cs.CV', 'cs.AI'],
    pdfUrl: 'https://arxiv.org/pdf/2401.02990',
    absUrl: 'https://arxiv.org/abs/2401.02990'
  },
  {
    id: '3',
    arxivId: '2311.03099',
    title: 'Efficient Fine-Tuning of Large Language Models for Domain-Specific Tasks',
    authors: ['Sarah Chen', 'Michael Liu', 'David Kim'],
    summary: 'Fine-tuning large language models for specific domains is computationally expensive and requires significant resources. In this work, we propose a novel efficient fine-tuning method that reduces computational costs by 90% while maintaining competitive performance. Our approach leverages parameter-efficient techniques and adaptive learning rates to achieve state-of-the-art results on various domain-specific benchmarks.',
    publishedAt: '2023-11-05',
    categories: ['cs.LG', 'cs.AI'],
    pdfUrl: 'https://arxiv.org/pdf/2311.03099',
    absUrl: 'https://arxiv.org/abs/2311.03099',
    label: 'disliked'
  },
  {
    id: '4',
    arxivId: '2402.01848',
    title: 'Graph Neural Networks for Molecular Property Prediction',
    authors: ['Anna Schmidt', 'Thomas Mueller', 'Lisa Weber'],
    summary: 'Graph neural networks (GNNs) have shown great promise in molecular property prediction tasks. This paper presents a novel GNN architecture that incorporates attention mechanisms and graph pooling strategies to capture complex molecular structures effectively. Experimental results on multiple benchmark datasets demonstrate significant improvements over existing methods.',
    publishedAt: '2024-02-03',
    categories: ['cs.LG', 'q-bio.QM'],
    pdfUrl: 'https://arxiv.org/pdf/2402.01848',
    absUrl: 'https://arxiv.org/abs/2402.01848'
  },
  {
    id: '5',
    arxivId: '2310.17681',
    title: 'Reinforcement Learning for Autonomous Navigation in Complex Environments',
    authors: ['James Park', 'Emily Davis', 'Robert Taylor'],
    summary: 'Autonomous navigation in complex, dynamic environments remains a challenging problem in robotics. We propose a reinforcement learning approach that combines deep Q-learning with hierarchical planning to enable robots to navigate efficiently in unstructured environments. Our method outperforms traditional navigation algorithms in terms of path efficiency and collision avoidance.',
    publishedAt: '2023-10-27',
    categories: ['cs.RO', 'cs.AI'],
    pdfUrl: 'https://arxiv.org/pdf/2310.17681',
    absUrl: 'https://arxiv.org/abs/2310.17681',
    label: 'liked'
  },
  {
    id: '6',
    arxivId: '2403.00567',
    title: 'Multi-Modal Large Language Models: Challenges and Opportunities',
    authors: ['Zhiqiang Zhang', 'Wei Li', 'Fang Wang'],
    summary: 'Multi-modal large language models represent the next frontier in AI research, enabling machines to process and understand multiple types of data simultaneously. This paper discusses the key challenges in building such models, including data integration, model architecture design, and training efficiency. We also highlight promising research directions and potential applications.',
    publishedAt: '2024-03-01',
    categories: ['cs.AI', 'cs.CV', 'cs.CL'],
    pdfUrl: 'https://arxiv.org/pdf/2403.00567',
    absUrl: 'https://arxiv.org/abs/2403.00567'
  },
  {
    id: '7',
    arxivId: '2309.12876',
    title: 'Federated Learning for Privacy-Preserving AI',
    authors: ['Maria Garcia', 'Juan Lopez', 'Carlos Ruiz'],
    summary: 'Federated learning enables training AI models on decentralized data while preserving privacy. This work presents a comprehensive framework for federated learning that addresses key challenges such as communication efficiency, model aggregation, and client heterogeneity. Experimental results demonstrate the effectiveness of our approach on various real-world datasets.',
    publishedAt: '2023-09-20',
    categories: ['cs.LG', 'cs.CR'],
    pdfUrl: 'https://arxiv.org/pdf/2309.12876',
    absUrl: 'https://arxiv.org/abs/2309.12876'
  },
  {
    id: '8',
    arxivId: '2401.15678',
    title: 'Self-Supervised Learning for Computer Vision: A Comprehensive Review',
    authors: ['Bin Yang', 'Lei Zhang', 'Xin Chen'],
    summary: 'Self-supervised learning has revolutionized computer vision by enabling models to learn from unlabeled data. This review provides an overview of self-supervised learning techniques, including contrastive learning, masked modeling, and generative approaches. We discuss their applications and compare their performance on various downstream tasks.',
    publishedAt: '2024-01-25',
    categories: ['cs.CV', 'cs.LG'],
    pdfUrl: 'https://arxiv.org/pdf/2401.15678',
    absUrl: 'https://arxiv.org/abs/2401.15678',
    label: 'liked'
  }
]

export const mockRecommendedPapers: RecommendedPaper[] = [
  {
    ...mockPapers[5],
    similarityScore: 0.87,
    reason: 'High similarity with your labeled papers on multi-modal learning'
  },
  {
    ...mockPapers[7],
    similarityScore: 0.79,
    reason: 'Related to your interest in computer vision and self-supervised learning'
  },
  {
    ...mockPapers[1],
    similarityScore: 0.73,
    reason: 'Matches your focus on vision-language models'
  },
  {
    id: '9',
    arxivId: '2404.01234',
    title: 'Advances in Large Language Model Alignment',
    authors: ['Alex Johnson', 'Rachel Adams', 'Sam Wilson'],
    summary: 'Aligning large language models with human values and intentions is crucial for safe and useful AI systems. This paper explores various alignment techniques, including reinforcement learning from human feedback (RLHF), constitutional AI, and iterative refinement methods. We discuss current challenges and propose directions for future research.',
    publishedAt: '2024-04-01',
    categories: ['cs.AI', 'cs.CL'],
    pdfUrl: 'https://arxiv.org/pdf/2404.01234',
    absUrl: 'https://arxiv.org/abs/2404.01234',
    similarityScore: 0.81,
    reason: 'High relevance to your previously labeled papers on AI alignment'
  },
  {
    id: '10',
    arxivId: '2404.05678',
    title: 'Efficient Transformers for Long Document Understanding',
    authors: ['Emma Thompson', 'James Brown', 'Lisa Anderson'],
    summary: 'Transformer architectures have become the backbone of NLP, but they struggle with long documents due to computational complexity. This work introduces a novel efficient transformer variant that reduces memory usage while maintaining performance on long document tasks. Our model achieves state-of-the-art results on document classification and summarization benchmarks.',
    publishedAt: '2024-04-10',
    categories: ['cs.CL', 'cs.AI'],
    pdfUrl: 'https://arxiv.org/pdf/2404.05678',
    absUrl: 'https://arxiv.org/abs/2404.05678',
    similarityScore: 0.75
  }
]

export const mockLabeledPapers: LabeledPaper[] = mockPapers
  .filter(p => p.label !== undefined && p.label !== null)
  .map(p => ({
    ...p,
    label: p.label!,
    labeledAt: new Date().toISOString()
  }))

export const mockStats = {
  totalPapers: 1247,
  labeledPapers: 45,
  todayNewPapers: 23,
  recommendedPapers: 15
}

export const categories = [
  { value: '', label: 'All Categories' },
  { value: 'cs.AI', label: 'Artificial Intelligence' },
  { value: 'cs.AR', label: 'Hardware Architecture' },
  { value: 'cs.CC', label: 'Computational Complexity' },
  { value: 'cs.CE', label: 'Computational Engineering, Finance, and Science' },
  { value: 'cs.CG', label: 'Computational Geometry' },
  { value: 'cs.CL', label: 'Computation and Language' },
  { value: 'cs.CR', label: 'Cryptography and Security' },
  { value: 'cs.CV', label: 'Computer Vision and Pattern Recognition' },
  { value: 'cs.CY', label: 'Computers and Society' },
  { value: 'cs.DB', label: 'Databases' },
  { value: 'cs.DC', label: 'Distributed, Parallel, and Cluster Computing' },
  { value: 'cs.DL', label: 'Digital Libraries' },
  { value: 'cs.DM', label: 'Discrete Mathematics' },
  { value: 'cs.DS', label: 'Data Structures and Algorithms' },
  { value: 'cs.ET', label: 'Emerging Technologies' },
  { value: 'cs.FL', label: 'Formal Languages and Automata Theory' },
  { value: 'cs.GL', label: 'General Literature' },
  { value: 'cs.GR', label: 'Graphics' },
  { value: 'cs.GT', label: 'Computer Science and Game Theory' },
  { value: 'cs.HC', label: 'Human-Computer Interaction' },
  { value: 'cs.IR', label: 'Information Retrieval' },
  { value: 'cs.IT', label: 'Information Theory' },
  { value: 'cs.LG', label: 'Machine Learning' },
  { value: 'cs.LO', label: 'Logic in Computer Science' },
  { value: 'cs.MA', label: 'Multiagent Systems' },
  { value: 'cs.MM', label: 'Multimedia' },
  { value: 'cs.MS', label: 'Mathematical Software' },
  { value: 'cs.NA', label: 'Numerical Analysis' },
  { value: 'cs.NE', label: 'Neural and Evolutionary Computing' },
  { value: 'cs.NI', label: 'Networking' },
  { value: 'cs.OH', label: 'Other Computer Science' },
  { value: 'cs.OS', label: 'Operating Systems' },
  { value: 'cs.PF', label: 'Performance' },
  { value: 'cs.PL', label: 'Programming Languages' },
  { value: 'cs.RO', label: 'Robotics' },
  { value: 'cs.SC', label: 'Symbolic Computation' },
  { value: 'cs.SD', label: 'Sound' },
  { value: 'cs.SE', label: 'Software Engineering' },
  { value: 'cs.SI', label: 'Social and Information Networks' },
  { value: 'cs.SY', label: 'Systems and Control' },
  { value: 'stat.AP', label: 'Statistics Applications' },
  { value: 'stat.CO', label: 'Computational Statistics' },
  { value: 'stat.ME', label: 'Methodology' },
  { value: 'stat.ML', label: 'Machine Learning' },
  { value: 'stat.OT', label: 'Other Statistics' },
  { value: 'stat.TH', label: 'Statistics Theory' },
  { value: 'physics.quant-ph', label: 'Quantum Physics' },
  { value: 'math.AP', label: 'Analysis of PDEs' },
  { value: 'math.CV', label: 'Complex Variables' },
  { value: 'math.GR', label: 'Group Theory' },
  { value: 'math.LO', label: 'Logic' },
  { value: 'math.PR', label: 'Probability' },
  { value: 'math.ST', label: 'Statistics Theory' },
  { value: 'q-bio', label: 'Quantitative Biology' },
  { value: 'q-fin', label: 'Quantitative Finance' },
  { value: 'econ', label: 'Economics' },
  { value: 'hep-th', label: 'High Energy Physics - Theory' },
  { value: 'hep-ph', label: 'High Energy Physics - Phenomenology' },
  { value: 'hep-ex', label: 'High Energy Physics - Experiment' },
  { value: 'gr-qc', label: 'General Relativity and Quantum Cosmology' },
  { value: 'astro-ph', label: 'Astrophysics' }
]
