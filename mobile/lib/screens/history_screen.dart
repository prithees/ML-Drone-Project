import 'package:flutter/material.dart';

import '../api_client.dart';
import '../theme.dart';
import 'history_map_screen.dart';

String _formatTimestamp(DateTime dt) {
  final local = dt.toLocal();
  final hh = local.hour.toString().padLeft(2, '0');
  final mm = local.minute.toString().padLeft(2, '0');
  return '${local.day}/${local.month}/${local.year} · $hh:$mm';
}

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  final _api = ApiClient();
  late Future<List<HistoryEntry>> _future;

  @override
  void initState() {
    super.initState();
    _future = _api.getHistory(limit: 200);
  }

  void _refresh() {
    setState(() => _future = _api.getHistory(limit: 200));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan History'),
        actions: [
          IconButton(
            icon: const Icon(Icons.map_outlined),
            tooltip: 'Map view',
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => const HistoryMapScreen()),
              );
            },
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: () async => _refresh(),
        child: FutureBuilder<List<HistoryEntry>>(
          future: _future,
          builder: (context, snapshot) {
            if (snapshot.connectionState != ConnectionState.done) {
              return const Center(child: CircularProgressIndicator());
            }
            if (snapshot.hasError) {
              return ListView(
                padding: const EdgeInsets.all(20),
                children: [
                  Text('Could not load history: ${snapshot.error}', style: const TextStyle(color: danger)),
                ],
              );
            }
            final entries = snapshot.data!;
            if (entries.isEmpty) {
              return ListView(
                padding: const EdgeInsets.all(20),
                children: const [
                  Padding(
                    padding: EdgeInsets.symmetric(vertical: 40),
                    child: Center(child: Text('No scans yet.', style: TextStyle(color: muted))),
                  ),
                ],
              );
            }
            return ListView.builder(
              padding: const EdgeInsets.all(20),
              itemCount: entries.length,
              itemBuilder: (context, i) => _entryTile(entries[i]),
            );
          },
        ),
      ),
    );
  }

  Widget _entryTile(HistoryEntry entry) {
    final IconData icon;
    final Color color;
    final String label;
    if (entry.sprayed) {
      icon = Icons.water_drop;
      color = warn;
      label = 'Sprayed';
    } else if (entry.matched) {
      icon = Icons.check_circle_outline;
      color = accent;
      label = 'Matched';
    } else {
      icon = Icons.circle_outlined;
      color = muted;
      label = 'No match';
    }

    return Card(
      child: ListTile(
        leading: Icon(icon, color: color),
        title: Text('$label · ${(entry.score * 100).toStringAsFixed(1)}%'),
        subtitle: Text(
          entry.hasLocation
              ? '${_formatTimestamp(entry.timestamp)} · ${entry.source} · ${entry.lat!.toStringAsFixed(5)}, ${entry.lng!.toStringAsFixed(5)}'
              : '${_formatTimestamp(entry.timestamp)} · ${entry.source}',
          style: const TextStyle(fontSize: 12),
        ),
      ),
    );
  }
}
