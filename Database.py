import streamlit as st
from utils import st_def
st_def.st_logo(title='👋', page_title="Welcome!",)

tab1, tab2, tab3 = st.tabs(["LLM", "General", "NLTK"])

with tab1:
    st.markdown("""
    for a conversational AI to be successful, it needs to meet three criteria:
    
    - It needs to generate human language/reasoning.
    - It needs to be able to **remember** what was said earlier to hold a proper conversation.
    - It needs to be able to **query factual information** outside of the general knowledge.
    
    While general-purpose LLMs can cover the first criterion, they need support for the other two. This is where vector databases can come into play:

    - Give LLMs state: LLMs are stateless. That means that once an LLM is trained, its knowledge is frozen. Although you can fine-tune LLMs to extend their knowledge with further information, once the fine-tuning is done, the LLM is in a frozen state again. Vector databases can effectively give LLMs state because you can easily create and update the information in a vector database.
    - Act as an external knowledge database: LLMs, like GPT-3, generate confident-sounding answers independently of their factual accuracy. Especially if you move outside of the general knowledge into domain-specific areas where the relevant facts may not have been a part of the training data, they can start to “hallucinate” (a phenomenon where LLMs generate factually incorrect answers). To combat hallucinations, you can use a vector search engine to retrieve the relevant factual knowledge and pipe it into the LLM’s context window. This practice is known as retrieval-augmented generation (RAG) and helps LLMs generate factually accurate results.
    """)

with tab2:
    st.image("./images/qdrant.png")
    st.image("./images/chroma2.png")
    st.image("./images/pinecone.png")
    st.image("./images/weaviate.png")
    st.image("./images/faiss.png")
    st.image("./images/vectordb.png")

    st.markdown("""
         🚀 Template-Based OCR (Optical Character Recognition) 🍨
             📄Rule-Based Text Extraction📚: 🔍 Python Libraries for Traditional Machine Learning Approaches📰
        🍨 Text Extraction App Using Streamlit and OpenAI Vision
        尚不能称之为向量数据库的 FAISS，玩票性质的 redisearch 和 pgvector，闭源的 SAAS 服务 pinecone，以及使用 Rust 构建的 qdrant 和 lancedb。这些向量数据库各有千秋，支持的索引技术不尽相同，但它们都试图解决传统数据库或者搜索引擎在搜索高维度信息时的力不从心的问题。
    
    我们知道，传统的关系型数据库擅长结构化数据存储，支持复杂条件下的精确匹配，但在搜索信息上和搜索引擎类似，通过前缀树，倒排索引，N-gram 索引或者类似的机制找到包含特定标记的文档，然后使用一个评分算法（如 TF-IDF，BM25）对匹配的结果排序。这种搜索强于浅层的文本匹配，但无法理解同一文本在不同表达下的语义内容或者不同上下文下拥有的语境关系。此外，对于图形，声音这样的比文本更高维度的内容，传统的方式就完全无法处理了。要想高效地检索或者匹配这样的数据，我们就要想办法为它们找到一种更好地数学表达方式，这就要用到机器学习中大家耳熟能详的 embedding 了。
    
    比如植根于 postgres 的 pgvector，虽然它提供了向量存储和索引，可以很方便地用 SQL 进行向量的相似性匹配，但其在处理大规模向量时依旧有很大的性能问题和存储开销。除了 pgvector，我还用过  redisearch，它在 embedding 数量仅仅在百万级规模时，就要花巨量的内存（> 16G），以及非常漫长的构建索引的时间（没有具体 benchmark，目测比 qdrant 慢了一到两个数量级）。
    
    对于少量的 embedding，比如几千到几十万这样的量级，我们可以用一门支持 SIMD 处理的高效语言（比如 Rust），辅以缓存机制，直接遍历计算即可，没必要引入向量数据库。但当 embedding 的数量很多时，我们就需要借助向量数据库来查询。此刻，暴力计算已经无法满足基本的延迟需求了。所以，我们需要引入多种索引方式来提升查询的效率，这是向量数据库所擅长的。以下是一些大家常用的索引技术和方法：
    
    ### 如何选择合适的向量数据库？
    选择合适的向量数据库需要考虑多个因素，因为不同的应用和场景可能对性能、可扩展性、持久性和其他功能有不同的需求。以下是在选择向量数据库时需要考虑的关键因素：
    
    1. 数据规模与查询速度：如果您的应用中有大量的数据需要索引，那么需要一个能够有效处理大规模数据的数据库。此外，查询速度是另一个关键指标，特别是对于实时应用。
    2. 搜索准确性：根据应用的需求，评估近似搜索的准确性。有些方法可能更快，但牺牲了一定的准确性。目前在我个人的使用中，我还没有发现在搜索准确性上，不同的向量数据库，如 qdrant，lancedb，redisearch 等有明显的区别。
    3. 索引构建时间：有些向量数据库在处理大量数据时，其索引构建过程可能会很慢。根据应用的需求，评估这是否是一个关键因素。前文说过，在百万向量这个级别，我发现 redisearch 构建索引的速度就明显低于 `qdrant` 一个甚至多个量级。
    4. 特性和功能：考虑其他功能，如数据更新、删除，重建索引，过滤等。我发现，很多时候，对复杂的应用，传统数据库的过滤功能是一个很有用的特性，比如，你可以机遇向量相似性来搜索类似的商品，同时你想限制搜索结果仅仅显示价格在特定区间并且有库存的商品。在我实验过的几个数据库中，貌似 qdrant 对 `filter`支持地最好。
    5. 开发和社区支持：选择一个有活跃社区和良好文档的数据库可以节省大量的开发时间，并在遇到问题时获得帮助。我个人一般会避开非开源的或者 license 不友好的向量数据库，因此，我仅仅对 pinecone 做了小量实验后，就换用了其他的向量数据库。
    6. 灵活性与定制性：根据您的需求，评估数据库是否允许定制索引和查询策略，是否容易二次开发。因为我个人偏好 Rust，所以我也特别偏好于用 Rust 撰写的 qdrant 和 lancedb。它们都提供了 Rust SDK/API，并且在我需要的时候，我可以修改其源码来满足我的特定需求。
    7. 集成与兼容性：考虑如何将数据库集成到现有的技术堆栈中。它是否支持您喜欢的编程语言？是否有REST API，GRPC 或其他方式与现有系统集成？由于目前向量数据库的主要使用场景还是跟 AI/ML 强相关，所以大多数向量数据库都会优先提供 python 的 SDK。对于其他语言的开发者而言，可能只能自己撰写或者依赖第三方提供的 SDK。
    8. 成本：评估总体成本，包括服务器、存储和开发时间。在这个层面，我觉得 **lancedb 很有前途**，它是目前唯一可用的支持无服务器部署的向量数据库 —— 也就是说，你可以将 lancedb 嵌入到你的应用程序中，甚至访问 s3 来获取数据。lancedb 官网上有一篇关于使用 aws lambda function 访问 s3 上的 db 进行语义搜索的例子，我没有亲测，但从 lancedb 的 dataset layout 上来看，这是可行的，唯一的不确定性在于延迟有多大。但无论如何，如果你的整个技术栈放在无服务器上，又要提供语义搜索能力，lancedb 是目前最好的选择。我个人认为无服务器将会是数据库的一个重大方向，很开心看到 neon 在无服务关系新数据库上的崛起；同样的，我也希望 lancedb 能在无服务向量数据库上打下一片天地。
    
    ### 纯载体数据库的优点
    
    利用索引技术进行高效的相似性搜索
    大型数据集和高查询工作负载的可扩展性
    支持高维数据
    **支持基于 HTTP 和 JSON 的 API**
    对向量运算的本机支持，包括加法、减法、点积、余弦相似度
    纯载体数据库的缺点
    
    仅矢量：纯矢量数据库可以存储矢量和一些元数据，但仅此而已。对于大多数企业人工智能用例，您可能需要包括实体、属性和层次结构（图形）、位置（地理空间）等的描述等数据。
    有限或没有 SQL 支持：纯向量数据库通常使用自己的查询语言，这使得很难对向量和相关信息运行传统分析，或者将向量和其他数据类型结合起来。
    没有完整的 CRUD。纯向量数据库并不是真正为创建、更新和删除操作而设计的。对于读取操作，数据必须首先进行矢量化和索引以进行持久化和检索。这些数据库专注于提取矢量数据、对其进行索引以进行有效的相似性搜索以及基于矢量相似性查询最近邻居。
    建立索引非常耗时。索引矢量数据计算量大、成本高且耗时。这使得很难将新数据用于生成人工智能应用程序。
    被迫权衡。根据所使用的索引技术，矢量数据库要求客户在准确性、效率和存储之间进行权衡。例如，Pinecone 的 IMI 索引（反向多重索引，ANN 的一种变体）会产生存储开销，并且计算量很大。它主要针对静态或半静态数据集而设计，如果频繁添加、修改或删除向量，则可能会受到挑战。Milvus 使用称为“产品量化”和“分层可导航小世界”(HNSW) 的索引，这些索引是权衡搜索准确性和效率的近似技术。此外，其索引需要配置各种参数，使用不正确的参数选择可能会影响搜索结果的质量或导致效率低下。
    
    企业特征值得怀疑。许多矢量数据库在基本功能上严重落后，包括 ACID 事务、灾难恢复、RBAC、元数据过滤、数据库可管理性、可观察性等。这可能会导致严重的业务问题 - 类似于丢失所有数据的客户。
    对于许多客户来说，矢量数据库的局限性将归结为性价比。鉴于矢量运算的计算量大，OSS矢量数据库或矢量库成为特别大规模应用程序的可行替代方案。
    
    Pinecone
    
    优点：非常容易上手（无需托管负担，完全云原生），不需要用户了解向量化或向量索引的任何知识。根据他们的文档（也非常好），它只是工作。 缺点：完全专有，无法了解其内部运作和路线图，除非能够在GitHub上跟踪他们的进展。此外，某些用户的经验突显了依赖完全外部的第三方托管服务以及开发者在数据库设置和运行方面完全缺乏控制的危险。从长远来看，依赖完全托管的闭源解决方案的成本影响可能是显著的，考虑到存在大量的开源、自托管的替代方案。 我的看法：在2020-21年，当向量数据库还不太为人所知时，Pinecone在提供方便的开发者功能方面领先于其他供应商。快进到2023年，坦率地说，Pinecone现在提供的功能其他供应商也有，而且大多数其他供应商至少提供自托管、托管或嵌入式模式，更不用说他们的算法和底层技术的源代码对最终用户是透明的了。
    
    Weaviate
    
    优点：令人惊叹的文档（包括技术细节和持续实验），Weaviate似乎专注于构建最好的开发者体验，并且通过Docker非常容易上手。在查询方面，它能够快速产生亚毫秒级的搜索结果，并提供关键字和向量搜索功能。 缺点：由于Weaviate是使用Golang构建的，可扩展性是通过Kubernetes实现的，这种方法在数据变得非常大时需要大量的基础设施资源（与Milvus类似）。Weaviate的完全托管服务的长期成本影响尚不清楚，可能需要将其性能与其他基于Rust的替代方案（如Qdrant和LanceDB）进行比较（尽管时间将告诉我们哪种方法在最具成本效益的方式下扩展得更好）。 我的看法：Weaviate拥有一个强大的用户社区，开发团队正在积极展示极限可扩展性（数千亿个向量），因此它似乎面向的目标市场是拥有大量数据并希望进行向量搜索的大型企业。它提供关键字搜索和向量搜索，并且具有强大的混合搜索功能，可以适用于各种用例，直接与Elasticsearch等文档数据库竞争。Weaviate还积极关注数据科学和机器学习，通过向量数据库将其扩展到传统搜索和检索应用程序之外的领域。
    
    Qdrant
    
    优点：虽然Qdrant比Weaviate更新，但它也有很好的文档，可以帮助开发人员通过Docker轻松上手。它完全使用Rust构建，提供了开发人员可以通过其Rust、Python和Golang客户端访问的API，这些是目前后端开发人员最常用的语言。由于Rust的强大性能，它的资源利用似乎比使用Golang构建的替代品低（至少在我的经验中是如此）。目前，它通过分区和Raft共识协议实现可扩展性，这是数据库领域的标准做法。 缺点：作为相对较新的工具，Qdrant在查询用户界面等方面一直在迎头赶上Weaviate和Milvus等竞争对手，尽管这个差距在每个新版本中都在迅速缩小。 我的看法：我认为Qdrant有望成为许多公司首选的矢量搜索后端，这些公司希望最大限度地降低基础设施成本，并利用现代编程语言Rust的强大功能。在撰写本文时，混合搜索尚未可用，但根据他们的路线图，正在积极开发中。此外，Qdrant不断发布有关如何优化其HNSW实现（内存和磁盘上的实现）的更新，这将极大地帮助实现其长期的搜索准确性和可扩展性目标。很明显，Qdrant的用户社区正在迅速增长（有趣的是，比Weaviate的增长速度更快），根据其GitHub的星标历史记录！也许全世界都对Rust感到兴奋？无论如何，在我看来，在Qdrant上构建应用是非常有趣的😀。
    
    Milvus/Zilliz
    
    优点：作为向量数据库生态系统中存在时间较长的数据库，Milvus非常成熟，并提供了许多向量索引的选项。它完全使用Golang构建，具有极强的可扩展性。截至2023年，它是唯一一个提供可工作的DiskANN实现的主要供应商，据说这是磁盘上最高效的向量索引。缺点：在我看来，Milvus似乎是一个将可扩展性问题解决得非常彻底的解决方案-它通过代理、负载均衡器、消息代理、Kafka和Kubernetes的组合实现了高度可扩展性，这使得整个系统变得非常复杂和资源密集。客户端API（例如Python）也不像Weaviate和Qdrant等较新的数据库那样易读或直观，后者更注重开发者体验。我的看法：很明显，Milvus的构建理念是为了实现对向量索引的大规模可扩展性，而在许多情况下，当数据的大小不是太大时，Milvus可能会显得过于复杂。对于更静态和不频繁的大规模情况，Qdrant或Weaviate等替代方案可能更便宜且更快速地投入生产。
    
    Chroma
    
    优点：Chroma为开发人员提供了方便的Python/JavaScript接口，可以快速启动向量存储。它是市场上第一个默认提供嵌入模式的向量数据库，其中数据库和应用层紧密集成，使开发人员能够快速构建、原型设计和展示他们的项目。 缺点：与其他专门构建的供应商不同，Chroma主要是一个围绕现有的OLAP数据库（Clickhouse）和现有的开源向量搜索实现（hnswlib）的Python/TypeScript封装。目前（截至2023年6月），它没有实现自己的存储层。 我的看法：向量数据库市场正在快速发展，Chroma似乎倾向于采取“等待观望”的策略，是为数不多的旨在提供多种托管选项的供应商之一：无服务器/嵌入式、自托管（客户端-服务器）和云原生分布式SaaS解决方案，可能同时支持嵌入式和客户端-服务器模式。根据他们的路线图，Chroma的服务器实现正在进行中。Chroma带来的另一个有趣的创新领域是量化“查询相关性”，即返回结果与用户输入查询的接近程度。在他们的路线图中还列出了可视化嵌入空间，这是一个创新领域，可以使数据库在搜索之外的许多应用中使用。然而，从长远来看，我们还没有看到嵌入式数据库架构在向量搜索领域成功实现商业化，因此它的发展（以及下面描述的LanceDB）将是一个有趣的观察对象！
    
    LanceDB
    
    优点：LanceDB专为多模态数据（图像、音频、文本）的分布式索引和搜索而设计，构建在Lance数据格式之上，这是一种创新的、用于机器学习的新型列式数据格式。与Chroma一样，LanceDB使用嵌入式、无服务器架构，并且完全使用Rust从头开始构建，因此与Qdrant一起，这是仅有的另一个利用Rust的速度、内存安全性和相对较低资源利用率的主要向量数据库供应商。 缺点：LanceDB是一个非常年轻的数据库，因此许多功能正在积极开发中，并且由于工程团队规模较小，功能的优先级排序将是一个挑战。 我的看法：我认为在所有的向量数据库中，LanceDB与其他数据库的区别最大。这主要是因为它在数据存储层（使用Lance，一种比parquet更快速的新型列式格式，专为非常高效的查找而设计）和基础架构层面进行了创新-通过使用无服务器架构。因此，大大减少了许多基础架构的复杂性，极大地增加了开发人员构建直接连接到数据湖的分布式语义搜索应用程序的自由和能力。
    
    Vespa
    
    优点：提供了最“企业级就绪”的混合搜索能力，将关键字搜索和自定义向量搜索与HNSW相结合。尽管其他供应商如Weaviate也提供关键字和向量搜索，但Vespa是最早推出这种功能的供应商之一，这给他们足够的时间来优化其功能，使其快速、准确和可扩展。 缺点：与使用性能导向语言（如Go或Rust）编写的更现代的替代方案相比，开发人员体验不够流畅，这是由于应用层是用Java编写的。此外，直到最近，它并没有提供非常简单的设置和拆除开发实例的方法，例如通过Docker和Kubernetes。 我的看法：Vespa确实提供了非常好的功能，但它的应用程序主要是用Java编写的，而后端和索引层是用C++构建的。这使得随着时间的推移，它更难以维护，并且相对于其他替代方案而言，它的开发人员友好度较低。现在大多数新的数据库都是完全用一种语言编写的，通常是Golang或Rust，并且似乎在Weaviate、Qdrant和LanceDB等数据库中算法和架构的创新速度更快。
    
    Vald
    
    优点：通过高度分布式的架构，设计用于处理多模态数据存储，同时具有索引备份等有用功能。使用非常快速的ANN搜索算法NGT（邻域图和树），当与高度分布式的向量索引结合使用时，它是最快的ANN算法之一。 缺点：与其他供应商相比，Vald似乎没有那么多的关注度和使用量，并且文档没有明确描述使用了什么向量索引（“分布式索引”相当模糊）。它似乎完全由一个实体Yahoo! Japan资助，很少有关于其他主要用户的信息。 我的看法：我认为Vald是一个比其他供应商更为专业的供应商，主要满足Yahoo! Japan的搜索需求，并且整体上拥有一个更小的用户社区，至少根据他们在GitHub上的星标来看是如此。其中一部分原因可能是它总部位于日本，并且没有像其他在欧盟和湾区的供应商那样进行大规模的市场推广。
    
    Elasticsearch, Redis and pgvector
    
    优点：如果已经在使用现有的数据存储，如Elasticsearch、Redis或PostgreSQL，那么利用它们的向量索引和搜索功能是相当简单的，无需使用新技术。 缺点：现有的数据库不一定以最优的方式存储或索引数据，因为它们被设计为通用目的，结果是，在涉及百万级向量搜索及以上规模的数据时，性能会受到影响。Redis VSS（Vector Search Store）之所以快速，主要是因为它完全在内存中，但一旦数据超过内存大小，就需要考虑替代解决方案。 我的看法：我认为专为特定目的构建的向量数据库将逐渐在需要语义搜索的领域中与已有数据库竞争，主要是因为它们在向量搜索的最关键组件-存储层面上进行了创新。HNSW和ANN算法等索引方法在文献中有很好的文档记录，大多数数据库供应商都可以推出自己的实现，但专为特定目的构建的向量数据库具有根据任务进行优化的优势（更不用说它们是用现代编程语言如Go和Rust编写的），出于可扩展性和性能的原因，从长远来看，它们很可能在这个领域获胜。
    
    结论:万亿规模的问题
    
    很难想象在历史上的任何时候，任何一种数据库能够吸引如此多的公众关注，更不用说风险投资生态系统了。向量数据库供应商（如Milvus、Weaviate）试图解决的一个关键用例是如何以最低的延迟实现万亿级向量搜索。这是一项极其困难的任务，考虑到当今通过流式处理或批处理传输的数据量，专为存储和查询性能进行优化的专用向量数据库最有可能在不久的将来突破这个障碍。 我将以观察到的历史数据库世界中最成功的商业模式作为结束，即首先开源代码（以便激发技术周围的热情社区），然后通过托管服务或云服务来商业化工具。嵌入式数据库在这个领域相对较新，尚不清楚它们在产品商业化和长期收入方面的成功程度。因此，可以推断出完全闭源的产品可能无法占据大部分市场份额-从长远来看，我直觉认为重视开发者生态系统和开发者体验的数据库有可能蓬勃发展，并且建立一个活跃的相信该工具的开源社区将比你想象的更重要！ 希望大家觉得这个总结有用！在接下来的文章中，我将总结向量数据库中的底层搜索和索引算法，并深入探讨技术细节。
    
    https://thedataquarry.com/posts/vector-db-1/
    
    向量数据库（Vector Database），也称为向量相似度搜索引擎或近似最近邻（ANN）搜索数据库，
    
    Standalone vector indices like FAISS (Facebook AI Similarity Search) can significantly improve the search and retrieval of vector embeddings, but they lack capabilities that exist in any database. Vector databases, on the other hand, are purpose-built to manage vector embeddings, providing several advantages over using standalone vector indices:

Data management: Vector databases offer well-known and easy-to-use features for data storage, like inserting, deleting, and updating data. This makes managing and maintaining vector data easier than using a standalone vector index like FAISS, which requires additional work to integrate with a storage solution.
Metadata storage and filtering: Vector databases can store metadata associated with each vector entry. Users can then query the database using additional metadata filters for finer-grained queries.
Scalability: Vector databases are designed to scale with growing data volumes and user demands, providing better support for distributed and parallel processing. Standalone vector indices may require custom solutions to achieve similar levels of scalability (such as deploying and managing them on Kubernetes clusters or other similar systems). Modern vector databases also use serverless architectures to optimize cost at scale.
Real-time updates: Vector databases often support real-time data updates, allowing for dynamic changes to the data to keep results fresh, whereas standalone vector indexes may require a full re-indexing process to incorporate new data, which can be time-consuming and computationally expensive. Advanced vector databases can use performance upgrades available via index rebuilds while maintaining freshness.
Backups and collections: Vector databases handle the routine operation of backing up all the data stored in the database. Pinecone also allows users to selectively choose specific indexes that can be backed up in the form of “collections,” which store the data in that index for later use.
Ecosystem integration: Vector databases can more easily integrate with other components of a data processing ecosystem, such as ETL pipelines (like Spark), analytics tools (like Tableau and Segment), and visualization platforms (like Grafana) – streamlining the data management workflow. It also enables easy integration with other AI related tooling like LangChain, LlamaIndex, Cohere, and many others..
Data security and access control: Vector databases typically offer built-in data security features and access control mechanisms to protect sensitive information, which may not be available in standalone vector index solutions. Multitenancy through namespaces allows users to partition their indexes fully and even create fully isolated partitions within their own index.
In short, a vector database provides a superior solution for handling vector embeddings by addressing the limitations of standalone vector indices, such as scalability challenges, cumbersome integration processes, and the absence of real-time updates and built-in security measures, ensuring a more effective and streamlined data management experience. 
     
    
    
    
        """)
    st.image("./images/zhang.gif")

