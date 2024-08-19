# -*- coding: utf-8 -*-
#
# @author awtestergit
# @description populate knowledge warehouse
#
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
from models.embed import IntfloatEmbeddingModel
from qdrantclient_vdb.qdrant_knowledge import KnowledgeWarehouse

def knowledge_strings():
    texts = [
        """Economic growth is likely to decelerate in 2024 as the effects of monetary policy take a broader toll and post-pandemic tailwinds fade.
We expect real GDP growth to walk the line between a slight expansion and contraction for much of next year, also known as a soft landing. After tracking to a better-than-expected 2.8% real GDP growth in 2023, we forecast a below-trend 0.7% pace of expansion in 2024. Among the major components of GDP, consumer spending is likely to rise at a more muted pace next year, while fiscal spending could swing from a positive contributor in 2023 to a modest drag. Notable drops in business investment and housing activity in 2023 set the foundation for improved performance in 2024, even if the outlook remains muted amid higher interest rates; 2023 strength in services sector is likely to soften.
""",
        """We assume the hiking cycle is over, leaving the Fed Funds on hold at 5.25%-5.5% until the middle of 2024.
If inflation continues its moderating trajectory over the coming quarters, we think it is likely the FOMC will start to slowly normalize policy rates near the midpoint of next year. We forecast 25 bps cuts at each meeting beginning in June, bringing the Fed Funds target range to 4.00%-4.25% at the end of 2024. Concurrently, quantitative tightening, the Fed’s balance sheet runoff program, is expected to be maintained at the same pace through 2024. At $95 billion per month, quantitative tightening is projected to remove approximately $1 trillion from the economy next year.
""",
        """The U.S. consumer could begin to bend, but not break.
There are numerous reasons to expect consumer spending growth to slow next year from its firm pace in 2023: diminished excess savings, plateauing wage gains, low savings rates and less pent-up demand. Additionally, the restart of student loan payments and uptick in subprime auto and millennial credit card delinquencies are emerging signs of stress for some consumers. On the flipside, household balance sheets and debt servicing levels remain healthy. Tight labor markets continue to support employment and therefore income levels. Considering the cross currents, we think consumer spending growth can stay positive overall in 2024, but at a lower rate than 2023.
""",
        """The larger-than-expected fiscal boost to the U.S. economy in 2023 could flip to a slight headwind in 2024.
The fiscal deficit roughly doubled to $1.84 trillion—7.4% of GDP—in fiscal 2023 from $950 billion in 2022. While the full extent of this year’s deficit expansion would not be considered stimulus in a classic sense, it is clear the federal government took in a lot less cash than it sent out. Looking to 2024, we expect the federal deficit to narrow to a still very large 5.9% of GDP, reflecting a bit of belt-tightening on the spending side partly offset by higher interest outlays on government debt.
""",
        """Labor markets are showing signs of normalization to end 2023; unemployment could drift higher in 2024 while remaining low in historical context.
Momentum in the job market is starting to wane with slowing payroll growth and modestly rising unemployment, as well as declining quit rates and temporary help. Increased labor force participation and elevated immigration patterns over the past year have added labor supply, while a shortening work week indicates moderating demand for labor. Considering the challenges to add and retain workers coming out of the pandemic, businesses could be more reluctant than normal to shed workers in a slowing economic environment. Even so, less hiring activity could be enough to cause the unemployment rate to tick up to the mid-4% area by the end of next year due to worker churn. Already slowing wage gains should slow further in the context of a softer labor market.
""",
        """Inflation trends are cooling, but likely to remain above the Fed’s 2% target through 2024.
After reaching a four-decade high in 2022, inflation on both a headline and core basis has moderated significantly in 2023. Some categories have seen more improvement than others. For example, core goods inflation dropped from a peak of 12.4% in February 2022 to 0% in October 2023. Progress on core services inflation, which includes the sticky shelter category, has been slower. After peaking at 7.3% in February 2023, core services inflation was still running an elevated 5.5% in October 2023. We expect moderating shelter inflation in 2024 as the lag in market rents pricing should catch up in the inflation readings. We forecast core PCE prices—the Fed’s preferred inflation metric—to rise 2.4% in 2024, down from 3.4% in 2023.
""",    """Housing sector activity has dropped 30%-40% over the past 18 months amid the surge in mortgage rates.
With housing affordability metrics at a 40-year low, combined with 75% of mortgages locked in at 4% or below, the U.S. housing market is effectively frozen. Real residential investment tumbled at a 12% seasonally adjusted annual rate over the past six quarters. Meanwhile, home values rose 6% in 2023—to near all-time highs—amid tight supply and historically low vacancies. Given the already large drop in recent years, we think the housing market is one area of the economy that could perform better in 2024 than in 2023, even if trends remain soft in the near term.
""",
        """Supply chain bottlenecks are mostly in the rearview, while global supply chain restructuring will take time.
Over the past year, as inventory constraints and shipping costs have fallen, supply chain considerations have shifted from short-term tactics to longer-term strategies of minimizing costs while ensuring resiliency. Legislation passed in 2022 including the CHIPS and Science Act and Inflation Reduction Act provides incentive for certain strategic industries—including semiconductors and renewables—to onshore production. This has resulted in rising business investment in high-tech manufacturing structures over the past year. Bigger picture, we expect global supply chain adjustments to continue at a conservative pace, as even the simplest changes are both costly and complex.
""",
        """Pressures on the commercial real estate sector are likely to intensify.
The higher-for-longer interest rate environment and challenges among small and regional banks are resulting in tightening of lending standards and slowing slow growth. This is occurring across all loan types, but most acutely for the commercial real estate sector, where small and regional banks have meaningful exposure. With nearly $550 billion of maturing commercial real estate debt over the next year, losses are expected to mount for lenders and investors. While we do not expect this to be a systemic issue, reduced lending activity and potential investor losses could be an economic headwind.
""",
        """Geopolitical risks will remain top of mind.
Elevated trade tensions with China, the ongoing Russia-Ukraine war and conflict in the Middle East all point to continued uncertainties and risks heading into 2024. While direct U.S. economic impact has been limited thus far, the larger risk is for a supply shock of a critical commodity or good—energy, food, semiconductors—that triggers significant market disruption. Next year’s U.S. presidential election could be more impactful than recent cycles on geopolitics given the backdrop of already elevated tensions.
""",
    ]
    return texts

if __name__ == '__main__':
    config_file = "config.json"
    g_config = {}
    with open(config_file) as f:
        g_config = json.load(f)

    # embedding
    embedding_model = None
    embedding_path = g_config['EMBEDDING']
    embedding_model = IntfloatEmbeddingModel(embedding_path)

    #vdb
    vdb_ip = g_config['VDBIP']
    vdb_port = g_config['VDBPORT']
    collection_name = g_config['VDBNAME']
    vdb_conf = g_config['VDBCONF']
    top_k = g_config['TOP']
    client = QdrantClient(vdb_ip, port=vdb_port)

    # create collection
    # check if collection exists
    response = client.get_collections()
    names = [desc.name for desc in response.collections]
    vector_size = embedding_model.EMBED_SIZE
    if collection_name not in names:#create
        client.create_collection(collection_name, vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE))

    kw_vdb = KnowledgeWarehouse(client, collection_name, embedding_model)
    
    texts = knowledge_strings()
    kw_vdb.add_knowledge_in_bulk(texts)