<template>
	<div class="background-container">
		<div class="areaView">
			<a-row>
				<a-col>
					<h1 class="main-title">财智通鉴</h1>
				</a-col>
			</a-row>
			<!-- <a-row>
				<a-col>
					<h3 >
						祝你早日实现财富自由!
						
					</h3>
				</a-col>
			</a-row> -->
			<a-row>
				<a-col>
					<router-link to="/chat/" class="start-button">快速体验</router-link>
				</a-col>
			</a-row>
		</div>
	</div>
	<!-- 这是一个基于Agent+RAG实现的财智通鉴问机器人，前端使用vue.js，
	后端通过langchain+agent+chroma实现对金融数据的向量化存储，
	最终前端和后端通过流式api进行数据交互。 -->
</template>

<style scoped>
.background-container {
  width: 100vw;
  height: 100vh;
  background: url('@/assets/background2.png') no-repeat center center;
  background-size: cover;
}

.areaView{
	width: 30%;
	height: 40%;
	//background-color: aqua;
	float: right;
	margin-top: 15%;
	//margin-right: 2%;
}
	
.main-title {
  font-size: 2rem; /* text-3xl */
  font-weight: bold;
  color: #1f2937; /* text-gray-900 */
  text-align: left;
}

.row-2{
	margin-top: 10%;
}

.row-3{
	margin-top: 10%;
}

.start-button {
	color: white;
	font-size: 2.125rem; /* text-lg */
	background-color: chocolate;
	margin-top: 10%;
}
</style>
