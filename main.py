import random
import matplotlib.pyplot as plt

class Conecta4:
    """
    Classe para representar o jogo Conecta 4 em um tabuleiro 5x5.
    Jogadores: 'X' (humano) e 'O' (IA com Min-Max).
    """
    def __init__(self):
        # Inicializa tabuleiro vazio e jogador atual
        self.tabuleiro = [[' ']*5 for _ in range(5)]
        self.jogador_atual = 'X'
        # Pesos para avaliação heurística da IA
        self.pesos = {
            'duplas': 1.0,
            'trincas': 5.0,
            'centro': 2.0,
            'potencial_vitoria': 10.0
        }

    def imprimir_tabuleiro(self):
        """
        Exibe o estado atual do tabuleiro no console.
        """
        for linha in self.tabuleiro:
            print('|'.join(linha))
        print('-' * 9)

    def acoes_possiveis(self):
        """
        Retorna lista de posições livres (linha, coluna).
        """
        return [(i, j)
                for i in range(5)
                for j in range(5)
                if self.tabuleiro[i][j] == ' ']

    def fazer_jogada(self, i, j):
        """
        Coloca a peça do jogador atual na posição (i, j) se estiver livre.
        Retorna True se OK, False caso contrário.
        """
        if self.tabuleiro[i][j] == ' ':
            self.tabuleiro[i][j] = self.jogador_atual
            return True
        return False

    def trocar_jogador(self):
        """
        Alterna entre 'X' e 'O'.
        """
        self.jogador_atual = 'O' if self.jogador_atual == 'X' else 'X'

    def verificar_vitoria(self):
        """
        Verifica todas as direções em busca de 4 peças consecutivas.
        Retorna 'X', 'O' ou None.
        """
        for i in range(5):
            for j in range(5):
                p = self.tabuleiro[i][j]
                if p != ' ':
                    # Horizontal
                    if j + 3 < 5 and all(self.tabuleiro[i][j + k] == p for k in range(4)):
                        return p
                    # Vertical
                    if i + 3 < 5 and all(self.tabuleiro[i + k][j] == p for k in range(4)):
                        return p
                    # Diagonal principal
                    if (i + 3 < 5 and j + 3 < 5 and
                        all(self.tabuleiro[i + k][j + k] == p for k in range(4))):
                        return p
                    # Diagonal secundária
                    if (i - 3 >= 0 and j + 3 < 5 and
                        all(self.tabuleiro[i - k][j + k] == p for k in range(4))):
                        return p
        return None

    def verificar_empate(self):
        """
        Retorna True se não houver espaços livres e nenhum vencedor.
        """
        return (all(self.tabuleiro[i][j] != ' '
                    for i in range(5)
                    for j in range(5))
                and self.verificar_vitoria() is None)

    def minmax(self, prof, alpha, beta, maxim):
        """
        Implementa Min-Max com poda alfa-beta.
        prof: profundidade restante
        alpha, beta: limites de poda
        maxim: True para IA (maximizador), False para humano (minimizador)
        Retorna (valor heurístico, melhor movimento).
        """
        win = self.verificar_vitoria()
        if prof == 0 or win or self.verificar_empate():
            return self.avaliar_tabuleiro(), None

        melhor_mov = None
        if maxim:
            aval = -float('inf')
            for i, j in self.acoes_possiveis():
                self.tabuleiro[i][j] = 'O'
                val, _ = self.minmax(prof - 1, alpha, beta, False)
                self.tabuleiro[i][j] = ' '
                if val > aval:
                    aval, melhor_mov = val, (i, j)
                alpha = max(alpha, aval)
                if beta <= alpha:
                    break  # poda beta
            return aval, melhor_mov
        else:
            aval = float('inf')
            for i, j in self.acoes_possiveis():
                self.tabuleiro[i][j] = 'X'
                val, _ = self.minmax(prof - 1, alpha, beta, True)
                self.tabuleiro[i][j] = ' '
                if val < aval:
                    aval, melhor_mov = val, (i, j)
                beta = min(beta, aval)
                if beta <= alpha:
                    break  # poda alfa
            return aval, melhor_mov

    def avaliar_tabuleiro(self):
        """
        Heurística de avaliação:
        - Vitória imediata: ±inf
        - Soma ponderada de duplas, trincas, controle do centro e potenciais de vitória.
        """
        if self.verificar_vitoria() == 'O':
            return float('inf')
        if self.verificar_vitoria() == 'X':
            return -float('inf')

        s = 0
        # Contagem de duplas e trincas
        s += self.pesos['duplas'] * (self.contar('O', 2) - self.contar('X', 2))
        s += self.pesos['trincas'] * (self.contar('O', 3) - self.contar('X', 3))
        # Controle do centro
        centro = [(2,2), (2,1), (2,3), (1,2), (3,2)]
        for cx, cy in centro:
            if self.tabuleiro[cx][cy] == 'O':
                s += self.pesos['centro']
            elif self.tabuleiro[cx][cy] == 'X':
                s -= self.pesos['centro']
        # Potenciais de vitória (3 em linha + 1 espaço)
        s += self.pesos['potencial_vitoria'] * (
            self.contar_pot('O') - self.contar_pot('X')
        )
        return s

    def contar(self, jog, tam):
        """
        Conta sequências exatas de tamanho `tam` para `jog`.
        """
        dirs = [(0,1), (1,0), (1,1), (-1,1)]
        cnt = 0
        for i in range(5):
            for j in range(5):
                for dx, dy in dirs:
                    ok = True
                    for k in range(tam):
                        x, y = i + dx*k, j + dy*k
                        if not (0 <= x < 5 and 0 <= y < 5 and self.tabuleiro[x][y] == jog):
                            ok = False
                            break
                    if ok:
                        cnt += 1
        return cnt

    def contar_pot(self, jog):
        """
        Conta padrões com 3 peças de `jog` e 1 espaço vazio em sequência de 4.
        """
        dirs = [(0,1), (1,0), (1,1), (-1,1)]
        cnt = 0
        for i in range(5):
            for j in range(5):
                for dx, dy in dirs:
                    seq = vazio = 0
                    for k in range(4):
                        x, y = i + dx*k, j + dy*k
                        if not (0 <= x < 5 and 0 <= y < 5):
                            break
                        if self.tabuleiro[x][y] == jog:
                            seq += 1
                        elif self.tabuleiro[x][y] == ' ':
                            vazio += 1
                        else:
                            break
                    if seq == 3 and vazio == 1:
                        cnt += 1
        return cnt


class AlgoritmoGenetico:
    """
    Otimiza os pesos da IA (Conecta4) via Algoritmo Genético.
    - Seleção por torneio round-robin
    - Crossover em ponto único
    - Mutação aleatória
    - Monitoramento de melhor, pior e média de fitness
    """
    def __init__(self, pop_size, c_rate, m_rate, gens, num_p=10):
        # Hiperparâmetros do AG
        self.pop_size = pop_size     # número de indivíduos
        self.c_rate   = c_rate       # taxa de crossover
        self.m_rate   = m_rate       # taxa de mutação
        self.gens     = gens         # número de gerações
        self.num_p    = num_p        # partidas para fitness padrão
        # População inicial: pesos aleatórios
        self.pop = [
            {k: random.uniform(0,5)
             for k in ['duplas','trincas','centro','potencial_vitoria']}
            for _ in range(pop_size)
        ]
        # Histórico para plotagem
        self.hist = {'best': [], 'worst': [], 'mean': []}

    def sim(self, pO, pX):
        """
        Simula uma partida Min-Max vs Min-Max entre pO (IA) e pX.
        Retorna +1 (O vence), -1 (X vence) ou 0 (empate).
        """
        jogo = Conecta4()
        jogo.pesos = pO.copy()
        player = 'O'
        while True:
            val, mv = jogo.minmax(4, -float('inf'), float('inf'), player == 'O')
            if mv:
                jogo.fazer_jogada(*mv)
            win = jogo.verificar_vitoria()
            if win or jogo.verificar_empate():
                break
            # Troca de jogador e pesos
            player = 'X' if player == 'O' else 'O'
            jogo.pesos = pO.copy() if player == 'O' else pX.copy()
        return 1 if win == 'O' else (-1 if win == 'X' else 0)

    def default_fit(self, ind):
        """
        Fitness padrão: média de resultados contra IA com pesos fixos.
        """
        default = {'duplas':1,'trincas':5,'centro':2,'potencial_vitoria':10}
        soma = 0
        for _ in range(self.num_p):
            soma += self.sim(ind, default)
        return soma / self.num_p

    def tournament(self):
        """
        Round-robin interno: cada par de indivíduos joga uma vez.
        Retorna lista de fitness normalizados.
        """
        fit = [0] * self.pop_size
        for i in range(self.pop_size):
            for j in range(i+1, self.pop_size):
                res = self.sim(self.pop[i], self.pop[j])
                fit[i] += res
                fit[j] -= res
        denom = max(self.pop_size - 1, 1)
        return [f/denom for f in fit]

    def evolve(self):
        """
        Executa o loop de evolução:
        - Avalia fitness inicial
        - Para cada geração: seleção por roleta, crossover, mutação, reavaliação
        - Armazena métricas e plota ao final
        """
        try:
            ft = self.tournament()
            self._log(0, ft)
            for g in range(1, self.gens + 1):
                # Pesos para seleção proporcional
                weights = [max(f, 0) + 1e-6 for f in ft]
                new_pop = []
                # Gera nova população em pares
                for _ in range(self.pop_size // 2):
                    p1, p2 = random.choices(self.pop, weights=weights, k=2)
                    c1, c2 = self.cross(p1, p2)
                    new_pop.extend([self.mut(c1), self.mut(c2)])
                self.pop = new_pop
                ft = self.tournament()
                self._log(g, ft)
        except KeyboardInterrupt:
            print("Interrompido pelo usuário. Gerando gráfico...")
        finally:
            self._plot()

    def _log(self, gen, ft):
        """
        Registra e imprime estatísticas de geração, incluindo os pesos do melhor indivíduo.
        """
        best = max(ft)
        worst = min(ft)
        mean = sum(ft) / len(ft)
        # Índice e pesos do melhor indivíduo
        idx_best = ft.index(best)
        best_weights = self.pop[idx_best]
        self.hist['best'].append(best)
        self.hist['worst'].append(worst)
        self.hist['mean'].append(mean)
        print(f"Gen {gen}: best {best:.2f}, worst {worst:.2f}, mean {mean:.2f}, weights {best_weights}")
        """
        Registra e imprime estatísticas de geração:
        - best: maior fitness
        - worst: menor fitness
        - mean: média de fitness
        """
        best = max(ft)
        worst = min(ft)
        mean = sum(ft) / len(ft)
        self.hist['best'].append(best)
        self.hist['worst'].append(worst)
        self.hist['mean'].append(mean)
        print(f"Gen {gen}: best {best:.2f}, worst {worst:.2f}, mean {mean:.2f}")

    def cross(self, a, b):
        """
        Une dois pais via crossover de ponto único.
        """
        if random.random() < self.c_rate:
            keys = list(a.keys())
            pt = random.randint(1, len(keys) - 1)
            c1 = {k: (b if i < pt else a)[k] for i, k in enumerate(keys)}
            c2 = {k: (a if i < pt else b)[k] for i, k in enumerate(keys)}
            return c1, c2
        # Sem crossover: clonagem
        return a.copy(), b.copy()

    def mut(self, ind):
        """
        Aplica mutação aleatória em um coeficiente.
        """
        if random.random() < self.m_rate:
            key = random.choice(list(ind.keys()))
            ind[key] = random.uniform(0, 5)
        return ind

    def _plot(self):
        """
        Plota evolução de best, mean e worst ao longo das gerações.
        """
        gens = list(range(len(self.hist['mean'])))
        plt.figure()
        plt.plot(gens, self.hist['best'], label='Best')
        plt.plot(gens, self.hist['mean'], label='Mean')
        plt.plot(gens, self.hist['worst'], label='Worst')
        plt.title('Fitness Evolution')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    # Configuração do AG: população, taxas e iterações
    ag = AlgoritmoGenetico(
        pop_size = 10,
        c_rate   = 0.7,
        m_rate   = 0.3,
        gens     = 10,
        num_p    = 20
    )
    ag.evolve()
# if __name__ == '__main__':
#     # Configuração do AG: população, taxas e iterações
#     ag = AlgoritmoGenetico(
#         pop_size = 4,
#         c_rate   = 0.7,
#         m_rate   = 0.1,
#         gens     = 5,
#         num_p    = 5
#     )
#     ag.evolve()}