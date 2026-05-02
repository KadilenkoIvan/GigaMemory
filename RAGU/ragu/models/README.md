# Interface level

`LLM`, `Embedder` - interface level. Argument type for
various components. Example:

```
@dataclass
class MyCommunitySummarizer:
    fast_llm: LLM,
    heavy_llm: LLM,
    query_embedder: Embedder
    key_embedder: Embedder
```

# Network level

`ResponseCachingMixin`, `CachedAsyncOpenAI` work on network level.
They handle caching, retrying, rate limiting, and woring with
OpenAI API.

# Logic level

`LLMOpenAI`, `EmbedderOpenAI` implement `LLM` and `Embedder`
interfaces, using `CachedAsyncOpenAI` backend. Example:

```
client = CachedAsyncOpenAI()
summarizer = MyCommunitySummarizer(
    fast_llm=LLMOpenAI(client, 'qweno-tiny'),
    heavy_llm=LLMOpenAI(client, 'claude-mighty'),
    query_embedder=EmbedderOpenAI(client, 'bert', dim=768),
    key_embedder=EmbedderOpenAI(client, 'bert', dim=768),
)
```

Thus, rate limiting is shared between all models, since they use
the same client, caching and retrying is handled correctly.

# Example debugging

```
from ragu.common.logger import logger
logger.remove()
logger.add(sys.stdout, level="DEBUG") 

client = CachedAsyncOpenAI(
    base_url=os.environ['OPENAI_BASE_URL'],
    api_key=os.environ['OPENAI_API_KEY'],
    rate_min_delay=2,
    rate_max_simultaneous=10,
    retry_times_sec=(4, 8),
    cache='./llm_cache',
    debug_errors_storage='./llm_debug',
)
```

On error you can open the corresponding request:

```
from diskcache import Index
index = Index('./llm_debug')
info = index['1772078734557794672']
print(list(info['kwargs']))
print(info['kwargs']['output_schema'].model_json_schema())
print(info['kwargs']['conversation'][0]['content'])
```

And re-run it:

```
result = await client._uncached_raw_chat_completion(
    **info['kwargs'],
)
```